"""
PPO Training for Graph Layout Optimization
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from collections import deque
import time
from tqdm import tqdm

from src.enviroment import GraphLayoutEnv, MultiGraphEnv
from src.models import GNNPolicy, MLPPolicy


class RolloutBuffer:
    """Buffer to store rollout data for PPO."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.coords = []
        self.edge_indices = []
        self.actions_node = []
        self.actions_delta = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.num_nodes_list = []

    def add(self, coords, edge_index, action_node, action_delta, log_prob, reward, value, done, num_nodes):
        self.coords.append(coords)
        self.edge_indices.append(edge_index)
        self.actions_node.append(action_node)
        self.actions_delta.append(action_delta)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.num_nodes_list.append(num_nodes)

    def __len__(self):
        return len(self.rewards)


class PPOTrainer:
    """PPO Trainer for Graph Layout Optimization."""

    def __init__(
        self,
        policy: nn.Module,
        env,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_steps: int = 128,
        n_epochs: int = 4,
        batch_size: int = 64,
        device: str = "cpu",
    ):
        self.policy = policy.to(device)
        self.env = env
        self.device = device

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        self.buffer = RolloutBuffer()

    def compute_gae(self, rewards, values, dones, next_value):
        """Compute Generalized Advantage Estimation."""
        advantages = []
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = advantages + torch.tensor(values, dtype=torch.float32, device=self.device)

        return advantages, returns

    def collect_rollouts(self):
        """Collect rollout data from environment."""
        self.buffer.reset()

        obs, info = self.env.reset()
        graph_data = self.env.get_graph_data()

        for _ in range(self.n_steps):
            coords = torch.tensor(obs, dtype=torch.float32, device=self.device)
            edge_index = graph_data["edge_index"].to(self.device)

            with torch.no_grad():
                action, log_prob, value = self.policy.get_action(coords, edge_index)

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            self.buffer.add(
                coords=obs.copy(),
                edge_index=graph_data["edge_index"].clone(),
                action_node=action["node"],
                action_delta=action["delta"].copy(),
                log_prob=log_prob.item(),
                reward=reward,
                value=value.item(),
                done=done,
                num_nodes=self.env.num_nodes,
            )

            if done:
                obs, info = self.env.reset()
                graph_data = self.env.get_graph_data()
            else:
                obs = next_obs

        # Compute next value for GAE
        coords = torch.tensor(obs, dtype=torch.float32, device=self.device)
        edge_index = graph_data["edge_index"].to(self.device)
        with torch.no_grad():
            _, _, _, next_value = self.policy.forward(coords, edge_index)
            next_value = next_value.item()

        return next_value

    def update(self):
        """Update policy using PPO."""
        # Compute advantages
        next_value = self.collect_rollouts()
        advantages, returns = self.compute_gae(
            self.buffer.rewards,
            self.buffer.values,
            self.buffer.dones,
            next_value
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        total_loss = 0
        total_pg_loss = 0
        total_value_loss = 0
        total_entropy = 0

        indices = np.arange(len(self.buffer))

        for _ in range(self.n_epochs):
            np.random.shuffle(indices)

            for start in range(0, len(self.buffer), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                # Prepare batch (process one sample at a time for variable-size graphs)
                batch_log_probs = []
                batch_entropies = []
                batch_values = []

                for idx in batch_indices:
                    coords = torch.tensor(
                        self.buffer.coords[idx], dtype=torch.float32, device=self.device
                    )
                    edge_index = self.buffer.edge_indices[idx].to(self.device)
                    action_node = torch.tensor(
                        self.buffer.actions_node[idx], dtype=torch.long, device=self.device
                    )
                    action_delta = torch.tensor(
                        self.buffer.actions_delta[idx], dtype=torch.float32, device=self.device
                    )

                    # Forward pass
                    node_logits, graph_emb, node_embs, value = self.policy.forward(
                        coords, edge_index
                    )

                    # Node log prob
                    from torch.distributions import Categorical, Normal
                    node_dist = Categorical(logits=node_logits)
                    node_log_prob = node_dist.log_prob(action_node)
                    node_entropy = node_dist.entropy()

                    # Delta log prob
                    selected_node_emb = node_embs[action_node]
                    combined = torch.cat([selected_node_emb, graph_emb.squeeze(0)], dim=-1)
                    delta_mean = torch.tanh(self.policy.delta_head(combined))
                    delta_std = torch.exp(self.policy.delta_log_std)
                    delta_dist = Normal(delta_mean, delta_std)
                    delta_log_prob = delta_dist.log_prob(action_delta).sum()
                    delta_entropy = delta_dist.entropy().sum()

                    log_prob = node_log_prob + delta_log_prob
                    entropy = node_entropy + delta_entropy

                    batch_log_probs.append(log_prob)
                    batch_entropies.append(entropy)
                    batch_values.append(value.squeeze())

                batch_log_probs = torch.stack(batch_log_probs)
                batch_entropies = torch.stack(batch_entropies)
                batch_values = torch.stack(batch_values)

                batch_old_log_probs = torch.tensor(
                    [self.buffer.log_probs[i] for i in batch_indices],
                    dtype=torch.float32, device=self.device
                )
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # PPO loss
                ratio = torch.exp(batch_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_advantages
                pg_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.mse_loss(batch_values, batch_returns)
                entropy_loss = -batch_entropies.mean()

                loss = pg_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()
                total_pg_loss += pg_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += (-entropy_loss.item())

        n_updates = self.n_epochs * (len(self.buffer) // self.batch_size + 1)
        return {
            "loss": total_loss / n_updates,
            "pg_loss": total_pg_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
        }

    def train(self, total_timesteps: int, log_interval: int = 1, save_path: str = None):
        """
        Train the policy.

        Args:
            total_timesteps: Total number of environment steps
            log_interval: Print stats every N updates
            save_path: Path to save model checkpoints
        """
        n_updates = total_timesteps // self.n_steps
        episode_rewards = deque(maxlen=100)
        episode_improvements = deque(maxlen=100)

        obs, info = self.env.reset()
        episode_reward = 0

        print(f"Starting training for {total_timesteps} timesteps ({n_updates} updates)")
        print(f"Device: {self.device}")

        for update in tqdm(range(1, n_updates + 1), desc="Training"):
            # Update policy
            stats = self.update()

            # Track episode stats
            for i in range(len(self.buffer)):
                episode_reward += self.buffer.rewards[i]
                if self.buffer.dones[i]:
                    episode_rewards.append(episode_reward)
                    episode_reward = 0

            if update % log_interval == 0:
                avg_reward = np.mean(episode_rewards) if episode_rewards else 0
                print(f"\nUpdate {update}/{n_updates}")
                print(f"  Avg Episode Reward: {avg_reward:.2f}")
                print(f"  Loss: {stats['loss']:.4f}")
                print(f"  PG Loss: {stats['pg_loss']:.4f}")
                print(f"  Value Loss: {stats['value_loss']:.4f}")
                print(f"  Entropy: {stats['entropy']:.4f}")

            # Save checkpoint
            if save_path and update % (log_interval * 10) == 0:
                checkpoint_path = Path(save_path) / f"checkpoint_{update}.pt"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "policy_state_dict": self.policy.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "update": update,
                }, checkpoint_path)
                print(f"  Saved checkpoint to {checkpoint_path}")

        # Save final model
        if save_path:
            final_path = Path(save_path) / "final_model.pt"
            final_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.policy.state_dict(), final_path)
            print(f"Saved final model to {final_path}")

        return self.policy


def train_single_graph(
    graph_path: str,
    total_timesteps: int = 10000,
    device: str = "cpu",
    save_path: str = None,
):
    """Train on a single graph (per-graph optimization)."""
    env = GraphLayoutEnv(graph_path=graph_path)

    policy = GNNPolicy(
        input_dim=2,
        hidden_dim=128,
        num_gnn_layers=3,
    )

    trainer = PPOTrainer(
        policy=policy,
        env=env,
        device=device,
        n_steps=64,
        n_epochs=4,
        batch_size=32,
    )

    trainer.train(total_timesteps=total_timesteps, save_path=save_path)

    return policy, env


def train_multi_graph(
    graph_list_file: str,
    data_dir: str,
    total_timesteps: int = 100000,
    device: str = "cpu",
    save_path: str = None,
    hidden_dim: int = 256,
    move_scale: float = 0.1,
):
    """Train on multiple graphs (generalizable model)."""
    env = MultiGraphEnv(
        graph_list_file=graph_list_file,
        data_dir=data_dir,
        move_scale=move_scale,
    )

    policy = GNNPolicy(
        input_dim=2,
        hidden_dim=hidden_dim,
        num_gnn_layers=3,
    )

    trainer = PPOTrainer(
        policy=policy,
        env=env,
        device=device,
        n_steps=128,
        n_epochs=4,
        batch_size=64,
    )

    trainer.train(total_timesteps=total_timesteps, save_path=save_path)

    return policy


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="multi", choices=["single", "multi"])
    parser.add_argument("--graph", type=str, default=None, help="Path to single graph")
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_path", type=str, default="checkpoints")
    args = parser.parse_args()

    if args.mode == "single":
        if args.graph is None:
            args.graph = "data/rome/rome/grafo10000.38.graphml"
        train_single_graph(
            graph_path=args.graph,
            total_timesteps=args.timesteps,
            device=args.device,
            save_path=args.save_path,
        )
    else:
        train_multi_graph(
            graph_list_file="data/train_graph.txt",
            data_dir="data",
            total_timesteps=args.timesteps,
            device=args.device,
            save_path=args.save_path,
        )
