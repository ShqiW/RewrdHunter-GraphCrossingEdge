"""
Unified training module.

Handles:
- Environment creation based on args
- Model creation based on args
- PPO training loop
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import networkx as nx
from pathlib import Path
from collections import deque
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataclasses import asdict
from torch_geometric.data import Data, Batch

from src.tasks.base import BaseArgs


class RolloutBuffer:
    """Stores rollout data for PPO"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.node_features = []  # [x, y, degree]
        self.edge_indices = []
        self.edge_attrs = []  # [edge_length, is_crossing]
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def add(self, node_features, edge_index, edge_attr, action, log_prob, reward, value, done):
        self.node_features.append(node_features)
        self.edge_indices.append(edge_index)
        self.edge_attrs.append(edge_attr)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def __len__(self):
        return len(self.rewards)


def create_graph(args: BaseArgs) -> nx.Graph:
    """Create or load graph based on args"""
    graph_config = args.graph

    if graph_config.graph_path:
        graph = nx.read_graphml(graph_config.graph_path)
        graph = nx.convert_node_labels_to_integers(graph, ordering="sorted")
    else:
        graph = nx.erdos_renyi_graph(
            graph_config.num_nodes,
            graph_config.edge_prob,
            seed=args.seed
        )
        # Ensure connected
        if not nx.is_connected(graph):
            components = list(nx.connected_components(graph))
            for i in range(len(components) - 1):
                u = list(components[i])[0]
                v = list(components[i + 1])[0]
                graph.add_edge(u, v)

    return graph


def create_env(args: BaseArgs, graph: nx.Graph = None, graph_data=None):
    """Create environment based on args"""
    env_type = args.env.type

    match env_type:
        case "discrete":
            from src.envs.discrete import DiscreteGraphEnv
            if graph_data is not None:
                return DiscreteGraphEnv(graph_data=graph_data, config=args.env)
            else:
                return DiscreteGraphEnv(graph=graph, config=args.env)
        case _:
            raise NotImplementedError(f"Environment type {env_type} not implemented")


def create_model(args: BaseArgs, env):
    """Create model based on args"""
    model_type = args.model.type

    match model_type:
        case "gnn":
            from src.models.gnn import DiscreteGNNPolicy
            return DiscreteGNNPolicy(config=args.model)
        case _:
            raise NotImplementedError(f"Model type {model_type} not implemented")


class PPOTrainer:
    """PPO Trainer - unified for all tasks"""

    def __init__(
        self,
        policy: nn.Module,
        env,
        args: BaseArgs,
        dataset=None,  # RomeDataset for multi-graph training
    ):
        self.policy = policy.to(args.device)
        self.env = env
        self.args = args
        self.device = args.device
        self.dataset = dataset

        # PPO config
        ppo = args.ppo
        self.gamma = ppo.gamma
        self.gae_lambda = ppo.gae_lambda
        self.clip_eps = ppo.clip_eps
        self.entropy_coef = ppo.entropy_coef
        self.value_coef = ppo.value_coef
        self.max_grad_norm = ppo.max_grad_norm
        self.n_steps = ppo.n_steps
        self.n_epochs = ppo.n_epochs
        self.batch_size = ppo.batch_size

        self.optimizer = optim.Adam(policy.parameters(), lr=ppo.lr)
        self.buffer = RolloutBuffer()
        self.start_update = 0  # For resuming training

        self.history = {
            "episode_rewards": [],
            "episode_crossings": [],
            "episode_improvements": [],
            "losses": [],
            "entropies": [],
        }

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint to resume training"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.start_update = checkpoint["update"]
        self.history = checkpoint["history"]
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Resuming from update {self.start_update}")

    def compute_gae(self, rewards, values, dones, next_value):
        """Compute Generalized Advantage Estimation"""
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

    def _sample_new_env(self):
        """Sample a new graph from dataset and create environment"""
        if self.dataset is None:
            return self.env

        graph_data = self.dataset.sample()
        new_env = create_env(self.args, graph_data=graph_data)

        # Track sampled graphs for debugging
        if not hasattr(self, '_sampled_graphs'):
            self._sampled_graphs = []
        self._sampled_graphs.append((graph_data.graph_name, graph_data.num_nodes))

        return new_env

    def collect_rollouts(self):
        """Collect rollout data"""
        self.buffer.reset()

        # Sample new graph if using dataset
        if self.dataset is not None:
            self.env = self._sample_new_env()

        obs, info = self.env.reset()
        graph_data = self.env.get_graph_data()

        for _ in range(self.n_steps):
            # obs is now node_features [num_nodes, 3]
            node_features = torch.tensor(obs, dtype=torch.float32, device=self.device)
            edge_index = graph_data["edge_index"].to(self.device)
            edge_attr = graph_data["edge_attr"].to(self.device)

            with torch.no_grad():
                action, log_prob, value = self.policy.get_action(
                    node_features, edge_index, edge_attr
                )

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Update graph_data for edge features (they change with layout)
            graph_data = self.env.get_graph_data()

            self.buffer.add(
                node_features=obs.copy(),
                edge_index=graph_data["edge_index"].clone(),
                edge_attr=graph_data["edge_attr"].clone(),
                action=action,
                log_prob=log_prob.item(),
                reward=reward,
                value=value.item(),
                done=done,
            )

            if done:
                self.history["episode_crossings"].append(info.get("crossings", 0))
                self.history["episode_improvements"].append(info.get("improvement", 0))

                # Sample new graph from dataset
                if self.dataset is not None:
                    self.env = self._sample_new_env()

                obs, info = self.env.reset()
                graph_data = self.env.get_graph_data()
            else:
                obs = next_obs

        # Compute next_value for GAE
        node_features = torch.tensor(obs, dtype=torch.float32, device=self.device)
        edge_index = graph_data["edge_index"].to(self.device)
        edge_attr = graph_data["edge_attr"].to(self.device)
        with torch.no_grad():
            _, next_value, _ = self.policy.forward(node_features, edge_index, edge_attr)
            next_value = next_value.item()

        return next_value

    def update(self):
        """PPO update step with batched graph processing"""
        next_value = self.collect_rollouts()
        advantages, returns = self.compute_gae(
            self.buffer.rewards,
            self.buffer.values,
            self.buffer.dones,
            next_value
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0
        total_pg_loss = 0
        total_value_loss = 0
        total_entropy = 0

        indices = np.arange(len(self.buffer))

        for _ in range(self.n_epochs):
            np.random.shuffle(indices)

            for start in range(0, len(self.buffer), self.batch_size):
                end = min(start + self.batch_size, len(self.buffer))
                batch_indices = indices[start:end]

                # Create PyG Data objects for each sample
                data_list = []
                for idx in batch_indices:
                    data = Data(
                        x=torch.tensor(
                            self.buffer.node_features[idx],
                            dtype=torch.float32
                        ),
                        edge_index=self.buffer.edge_indices[idx],
                        edge_attr=self.buffer.edge_attrs[idx],
                    )
                    data_list.append(data)

                # Batch all graphs together
                batched_data = Batch.from_data_list(data_list).to(self.device)

                # Get actions for this batch
                batch_actions = torch.tensor(
                    [self.buffer.actions[i] for i in batch_indices],
                    dtype=torch.long, device=self.device
                )

                # Evaluate all graphs in one forward pass
                batch_log_probs, batch_entropies, batch_values = self.policy.evaluate_action_batched(
                    batched_data, batch_actions
                )

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

        self.history["losses"].append(total_loss / n_updates)
        self.history["entropies"].append(total_entropy / n_updates)

        return {
            "loss": total_loss / n_updates,
            "pg_loss": total_pg_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
        }

    def train(self):
        """Main training loop"""
        total_timesteps = self.args.total_timesteps
        log_interval = self.args.log_interval
        save_path = self.args.save_path

        n_updates = total_timesteps // self.n_steps
        recent_improvements = deque(maxlen=100)

        print(f"Starting PPO training: {self.args.name}")
        print(f"Action space: {self.env.num_actions} actions")
        print(f"Total timesteps: {total_timesteps}, Updates: {n_updates}")
        if self.start_update > 0:
            print(f"Resuming from update: {self.start_update}")
        print(f"Device: {self.device}")
        print("=" * 60)

        best_avg_improvement = float('-inf')

        for update in tqdm(range(self.start_update + 1, n_updates + 1), desc="Training", initial=self.start_update, total=n_updates):
            stats = self.update()

            if self.history["episode_improvements"]:
                recent_improvements.extend(self.history["episode_improvements"][-10:])

            if update % log_interval == 0:
                avg_improvement = np.mean(list(recent_improvements)) if recent_improvements else 0
                avg_crossings = np.mean(self.history["episode_crossings"][-100:]) if self.history["episode_crossings"] else 0

                if avg_improvement > best_avg_improvement:
                    best_avg_improvement = avg_improvement
                    marker = " *"
                else:
                    marker = ""

                print(f"\nUpdate {update}/{n_updates}{marker}")
                print(f"  Avg Improvement: {avg_improvement:.2f}")
                print(f"  Avg Final Crossings: {avg_crossings:.1f}")
                print(f"  Loss: {stats['loss']:.4f}, Entropy: {stats['entropy']:.3f}")

                # Show graph sampling stats
                if hasattr(self, '_sampled_graphs') and self._sampled_graphs:
                    unique_graphs = len(set(g[0] for g in self._sampled_graphs))
                    total_samples = len(self._sampled_graphs)
                    last_graph = self._sampled_graphs[-1]
                    print(f"  Graphs: {unique_graphs} unique / {total_samples} total, last: {last_graph[0]} ({last_graph[1]} nodes)")

            if save_path and update % (log_interval * 10) == 0:
                self._save_checkpoint(save_path, update)

        if save_path:
            self._save_checkpoint(save_path, n_updates, final=True)
            self._plot_training_curves(save_path)

        return self.policy

    def _save_checkpoint(self, save_path, update, final=False):
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        if final:
            path = save_dir / "final_model.pt"
        else:
            path = save_dir / f"checkpoint_{update}.pt"

        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "update": update,
            "history": self.history,
            "args": asdict(self.args),
        }, path)
        print(f"  Saved to {path}")

    def _plot_training_curves(self, save_path):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        def smooth(data, window=50):
            if len(data) < window:
                return data
            return np.convolve(data, np.ones(window)/window, mode='valid')

        if self.history["episode_improvements"]:
            ax = axes[0, 0]
            data = self.history["episode_improvements"]
            ax.plot(data, alpha=0.2, color='green')
            if len(data) >= 50:
                ax.plot(smooth(data), color='green', linewidth=2)
            ax.axhline(y=0, color='red', linestyle='--')
            ax.set_xlabel("Episode")
            ax.set_ylabel("Improvement")
            ax.set_title("Crossing Improvement")
            ax.grid(True)

        if self.history["episode_crossings"]:
            ax = axes[0, 1]
            data = self.history["episode_crossings"]
            ax.plot(data, alpha=0.2, color='blue')
            if len(data) >= 50:
                ax.plot(smooth(data), color='blue', linewidth=2)
            ax.set_xlabel("Episode")
            ax.set_ylabel("Crossings")
            ax.set_title("Final Crossings")
            ax.grid(True)

        if self.history["losses"]:
            ax = axes[1, 0]
            ax.plot(self.history["losses"])
            ax.set_xlabel("Update")
            ax.set_ylabel("Loss")
            ax.set_title("Training Loss")
            ax.grid(True)

        if self.history["entropies"]:
            ax = axes[1, 1]
            ax.plot(self.history["entropies"])
            ax.set_xlabel("Update")
            ax.set_ylabel("Entropy")
            ax.set_title("Policy Entropy")
            ax.grid(True)

        plt.tight_layout()
        plt.savefig(Path(save_path) / "training_curves.png", dpi=150)
        plt.close()


def train(args: BaseArgs):
    """
    Unified training entry point.

    1. Create dataset or single graph
    2. Create env based on args.env
    3. Create model based on args.model
    4. Run PPO training
    """
    print(f"Task: {args.name}")
    print(f"Seed: {args.seed}")

    dataset = None
    graph_data = None

    # Check if using dataset
    if args.graph.use_dataset:
        from src.data.rome import RomeDataset
        dataset = RomeDataset(
            root=args.graph.data_root,
            split=args.graph.data_split,
        )
        print(f"Dataset: {len(dataset)} graphs from {args.graph.data_split} split")

        # Sample initial graph
        graph_data = dataset.sample()
        print(f"Initial graph: {graph_data.graph_name} ({graph_data.num_nodes} nodes)")

        # Create environment from dataset sample
        env = create_env(args, graph_data=graph_data)
    else:
        # Single graph mode
        graph = create_graph(args)
        print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        env = create_env(args, graph=graph)

    obs, info = env.reset()
    print(f"Initial crossings: {info['crossings']:.1f}")

    # Create model
    model = create_model(args, env)
    print(f"Model: {args.model.type}, params: {sum(p.numel() for p in model.parameters())}")

    # Create trainer and train
    trainer = PPOTrainer(policy=model, env=env, args=args, dataset=dataset)

    # Load checkpoint if specified
    if args.load_checkpoint:
        trainer.load_checkpoint(args.load_checkpoint)

    trained_policy = trainer.train()

    return trained_policy, env, trainer.history
