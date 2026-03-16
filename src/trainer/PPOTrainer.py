# ──────────────────────────────────────────────────────────────────────────────
# PPO Trainer — discrete (original, unchanged)
# ──────────────────────────────────────────────────────────────────────────────

import torch.optim as optim
from pathlib import Path

import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataclasses import asdict

from src.data.RolloutBuffer import RolloutBuffer
from src.trainer.utils import create_env
from torch_geometric.data import Data, Batch

from investigation.logger.logger import DynamicCSVLogger


class PPOTrainer:
    """PPO Trainer for discrete action spaces."""

    def __init__(self, policy, env, device, args, dataset=None):
        self.policy = policy.to(device)
        self.env = env
        self.args = args
        self.device = device
        self.dataset = dataset

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

        self.logger = DynamicCSVLogger("history.csv")
        self.optimizer = optim.Adam(policy.parameters(), lr=ppo.lr)
        self.buffer = RolloutBuffer()
        self.start_update = 0

        self.history = {
            "episode_rewards": [],
            "episode_crossings": [],
            "episode_improvements": [],
            "episode_improvement_pcts": [],
            "losses": [],
            "entropies": [],
        }

    def load_checkpoint(self, checkpoint_path: Path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.start_update = checkpoint["global_step"]
        self.history = checkpoint["history"]
        self.logger.resync(self.start_update, "global_step")
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Resuming from update {self.start_update}")

    def compute_gae(self, rewards, values, dones, next_value):
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            next_val = next_value if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_val * (1 -
                                                          dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages,
                                  dtype=torch.float32,
                                  device=self.device)
        returns = advantages + torch.tensor(
            values, dtype=torch.float32, device=self.device)
        return advantages, returns

    def _sample_new_env(self):
        if self.dataset is None:
            return self.env
        graph_data = self.dataset.sample()
        new_env = create_env(
            self.args,
            graph_data=graph_data,
            device=self.device,
        )
        if not hasattr(self, '_sampled_graphs'):
            self._sampled_graphs = []
        self._sampled_graphs.append(
            (graph_data.graph_name, graph_data.num_nodes))
        return new_env

    def collect_rollouts(self):
        self.buffer.reset()
        rollout_reductions = []
        rollout_reduction_pcts = []

        if self.dataset is not None:
            self.env = self._sample_new_env()

        obs, info = self.env.reset()
        graph_data = self.env.get_graph_data()

        for _ in range(self.n_steps):
            node_features = torch.tensor(obs,
                                         dtype=torch.float32,
                                         device=self.device)
            edge_index = graph_data["edge_index"].to(self.device)
            edge_attr = graph_data["edge_attr"].to(self.device)

            with torch.no_grad():
                action, log_prob, value = self.policy.get_action(
                    node_features, edge_index, edge_attr)

            next_obs, reward, terminated, truncated, info = self.env.step(
                action)
            done = terminated or truncated
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
                improvement = info.get("improvement", 0)
                initial = info.get("initial_crossings", 0)
                rollout_reductions.append(improvement)
                rollout_reduction_pcts.append(improvement / initial *
                                              100 if initial > 0 else 0.0)
                self.history["episode_crossings"].append(
                    info.get("crossings", 0))
                self.history["episode_improvements"].append(improvement)
                self.history["episode_improvement_pcts"].append(
                    rollout_reduction_pcts[-1])
                if self.dataset is not None:
                    self.env = self._sample_new_env()
                obs, info = self.env.reset()
                graph_data = self.env.get_graph_data()
            else:
                obs = next_obs

        node_features = torch.tensor(obs,
                                     dtype=torch.float32,
                                     device=self.device)
        edge_index = graph_data["edge_index"].to(self.device)
        edge_attr = graph_data["edge_attr"].to(self.device)
        with torch.no_grad():
            _, next_value, _ = self.policy.forward(node_features, edge_index,
                                                   edge_attr)
            next_value = next_value.item()

        return next_value, rollout_reductions, rollout_reduction_pcts

    def update(self):
        next_value, rollout_reductions, rollout_reduction_pcts = self.collect_rollouts(
        )
        advantages, returns = self.compute_gae(self.buffer.rewards,
                                               self.buffer.values,
                                               self.buffer.dones, next_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() +
                                                         1e-8)
        ret_mean = returns.mean()
        ret_std = returns.std() + 1e-8

        total_loss = total_pg = total_vl = total_ent = 0
        indices = np.arange(len(self.buffer))

        for _ in range(self.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, len(self.buffer), self.batch_size):
                end = min(start + self.batch_size, len(self.buffer))
                bi = indices[start:end]

                data_list = [
                    Data(
                        x=torch.tensor(self.buffer.node_features[i],
                                       dtype=torch.float32),
                        edge_index=self.buffer.edge_indices[i],
                        edge_attr=self.buffer.edge_attrs[i],
                    ) for i in bi
                ]
                batched = Batch.from_data_list(data_list).to(self.device)
                batch_actions = torch.tensor(
                    [self.buffer.actions[i] for i in bi],
                    dtype=torch.long,
                    device=self.device)

                batch_log_probs, batch_entropies, batch_values = \
                    self.policy.evaluate_action_batched(batched, batch_actions)

                old_lp = torch.tensor([self.buffer.log_probs[i] for i in bi],
                                      dtype=torch.float32,
                                      device=self.device)
                batch_adv = advantages[bi]
                batch_ret_norm = (returns[bi] - ret_mean) / ret_std

                ratio = torch.exp(batch_log_probs - old_lp)
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps,
                                    1 + self.clip_eps) * batch_adv
                pg_loss = -torch.min(surr1, surr2).mean()
                vl = nn.functional.mse_loss(batch_values, batch_ret_norm)
                ent_loss = -batch_entropies.mean()

                loss = pg_loss + self.value_coef * vl + self.entropy_coef * ent_loss
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()
                total_pg += pg_loss.item()
                total_vl += vl.item()
                total_ent += (-ent_loss.item())

        n_updates = self.n_epochs * (len(self.buffer) // self.batch_size + 1)
        self.history["losses"].append(total_loss / n_updates)
        self.history["entropies"].append(total_ent / n_updates)
        return {
            "loss": total_loss / n_updates,
            "pg_loss": total_pg / n_updates,
            "value_loss": total_vl / n_updates,
            "entropy": total_ent / n_updates,
            "rollout_reductions": rollout_reductions,
            "rollout_reduction_pcts": rollout_reduction_pcts,
        }

    def train(self):
        total_timesteps = self.args.total_timesteps
        log_interval = self.args.log_interval
        save_path = self.args.save_path

        n_updates = total_timesteps // self.n_steps

        print(f"Starting PPO training: {self.args.name}")
        print(f"Action space: {self.env.num_actions} actions")
        print(f"Total timesteps: {total_timesteps}, Updates: {n_updates}")
        if self.start_update > 0:
            print(f"Resuming from update: {self.start_update}")
        print(f"Device: {self.device}")
        print("=" * 60)

        best_avg = float('-inf')

        for update in tqdm(
                range(self.start_update + 1, n_updates + 1),
                desc="Training",
                initial=self.start_update,
                total=n_updates,
        ):
            stats = self.update()

            if update % log_interval == 0:
                reductions = stats["rollout_reductions"]
                reduction_pcts = stats["rollout_reduction_pcts"]
                avg_reduction = np.mean(reductions) if reductions else float(
                    "nan")
                avg_reduction_pct = np.mean(
                    reduction_pcts) if reduction_pcts else float("nan")
                marker = " *" if (not np.isnan(avg_reduction)
                                  and avg_reduction > best_avg) else ""
                if not np.isnan(avg_reduction) and avg_reduction > best_avg:
                    best_avg = avg_reduction
                log_dict = {
                    "global_step": update,
                    "crossing_reduction": avg_reduction,
                    "crossing_reduction_pct": avg_reduction_pct,
                    "n_episodes": len(reductions),
                    "loss": stats["loss"],
                    "pg_loss": stats["pg_loss"],
                    "value_loss": stats["value_loss"],
                    "entropy": stats["entropy"],
                }
                self.logger.log(log_dict)
                print(f"\nUpdate {update}/{n_updates}{marker}")
                print(
                    f"  Crossing Reduction: {avg_reduction:.2f}  ({avg_reduction_pct:.1f}%)  [{len(reductions)} eps]"
                )
                print(f"  Loss: {stats['loss']:.4f}  "
                      f"pg={stats['pg_loss']:.4f}  "
                      f"vl={stats['value_loss']:.4f}  "
                      f"ent={stats['entropy']:.3f}")
                if hasattr(self, '_sampled_graphs') and self._sampled_graphs:
                    ug = len(set(g[0] for g in self._sampled_graphs))
                    tg = len(self._sampled_graphs)
                    last = self._sampled_graphs[-1]
                    print(
                        f"  Graphs: {ug} unique / {tg} total, last: {last[0]} ({last[1]} nodes)"
                    )

            if save_path and update % (log_interval * 10) == 0:
                self._save_checkpoint(save_path, update)

        if save_path:
            self._save_checkpoint(save_path, n_updates, final=True)
            self._plot_training_curves(save_path)

        return self.policy

    def _save_checkpoint(self, save_path, update, final=False):
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / ("final_model.pt"
                           if final else f"checkpoint_{update}.pt")
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "global_step": update,
                "history": self.history,
                "args": asdict(self.args),
            }, path)
        print(f"  Saved to {path}")

    def _plot_training_curves(self, save_path):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        def smooth(data, window=50):
            return np.convolve(data, np.ones(window) / window, mode='valid') \
                if len(data) >= window else data

        if self.history["episode_improvements"]:
            ax = axes[0, 0]
            d = self.history["episode_improvements"]
            ax.plot(d, alpha=0.2, color='green')
            if len(d) >= 50:
                ax.plot(smooth(d), color='green', linewidth=2)
            ax.axhline(y=0, color='red', linestyle='--')
            ax.set_xlabel("Episode")
            ax.set_ylabel("Improvement")
            ax.set_title("Crossing Improvement")
            ax.grid(True)

        if self.history["episode_crossings"]:
            ax = axes[0, 1]
            d = self.history["episode_crossings"]
            ax.plot(d, alpha=0.2, color='blue')
            if len(d) >= 50:
                ax.plot(smooth(d), color='blue', linewidth=2)
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
