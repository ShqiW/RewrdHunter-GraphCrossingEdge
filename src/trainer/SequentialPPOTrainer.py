import copy
import torch.optim as optim
import torch
from pathlib import Path
import numpy as np
import torch.nn as nn
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor

from src.trainer.utils import create_env
from src.data.ContinuousRolloutBuffer import ContinuousRolloutBuffer
from investigation.logger.logger import DynamicCSVLogger


class SequentialPPOTrainer:
    """
    PPO Trainer for SequentialGraphEnv + TransformerPlacementPolicy.

    Key differences from PPOTrainer:
    - Continuous 2D actions stored as float tensors
    - Variable-size node_features per step (no fixed graph topology)
    - PPO update uses padded batching inside the policy
    """

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
        self.n_envs = ppo.n_envs

        self.logger = DynamicCSVLogger("history.csv")

        self.optimizer = optim.Adam(policy.parameters(), lr=ppo.lr)
        self.buffer = ContinuousRolloutBuffer(ppo.n_steps, device)
        self.start_update = 0
        self._executor = ThreadPoolExecutor(
            max_workers=self.n_envs) if self.n_envs > 1 else None

        self.history = {
            "episode_crossings": [],  # total_crossings per episode (raw)
            "episode_crossing_rates":
            [],  # total_crossings / num_edges (normalized)
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
        new_env = create_env(self.args,
                             device=self.device,
                             graph_data=graph_data)
        if not hasattr(self, '_sampled_graphs'):
            self._sampled_graphs = []
        self._sampled_graphs.append(
            (graph_data.graph_name, graph_data.num_nodes))
        return new_env

    def collect_rollouts(self):
        self.buffer.reset()
        rollout_crossing_rates = []
        n_envs = self.n_envs
        n_ticks = self.n_steps // n_envs

        # Initialize N independent envs
        envs = []
        env_obs = []
        for _ in range(n_envs):
            env = self._sample_new_env() if self.dataset is not None \
                else copy.deepcopy(self.env)
            envs.append(env)
            obs, _ = env.reset()
            env_obs.append(obs)

        # Per-env step lists — kept separate for correct per-env GAE
        env_steps = [[] for _ in range(n_envs)]

        for _ in range(n_ticks):
            obs_tensors = [
                torch.tensor(obs, dtype=torch.float32, device=self.device)
                for obs in env_obs
            ]
            with torch.no_grad():
                actions, log_probs, values = self.policy.get_action_batched(
                    obs_tensors)

            # Parallel env stepping: submit all step() calls, collect results
            if self._executor is not None:
                futures = [
                    self._executor.submit(envs[i].step, actions[i])
                    for i in range(n_envs)
                ]
                step_results = [f.result() for f in futures]
            else:
                step_results = [
                    envs[i].step(actions[i]) for i in range(n_envs)
                ]

            # Done handling in main thread (shared state: history, envs list)
            for i, (next_obs, reward, terminated, truncated,
                    info) in enumerate(step_results):
                done = terminated or truncated

                env_steps[i].append(
                    dict(
                        node_features=obs_tensors[i],
                        action=actions[i].copy(),
                        log_prob=log_probs[i].item(),
                        reward=reward,
                        value=values[i].item(),
                        done=done,
                    ))

                if done:
                    total_xing = info["total_crossings"]
                    crossing_rate = total_xing / envs[i].num_edges \
                        if envs[i].num_edges > 0 else 0.0
                    rollout_crossing_rates.append(crossing_rate)
                    self.history["episode_crossings"].append(total_xing)
                    self.history["episode_crossing_rates"].append(
                        crossing_rate)
                    if self.dataset is not None:
                        envs[i] = self._sample_new_env()
                    env_obs[i], _ = envs[i].reset()
                else:
                    env_obs[i] = next_obs

        # Bootstrap: one batched forward pass for all N last observations
        last_obs_tensors = [
            torch.tensor(obs, dtype=torch.float32, device=self.device)
            for obs in env_obs
        ]
        with torch.no_grad():
            _, _, boot_values = self.policy.forward(last_obs_tensors)  # [N]

        # Per-env GAE → fill buffer in env order (indices stay consistent)
        all_advantages = []
        all_returns = []
        for i in range(n_envs):
            steps = env_steps[i]
            if not steps:
                continue
            adv_i, ret_i = self.compute_gae(
                [s["reward"] for s in steps],
                [s["value"] for s in steps],
                [s["done"] for s in steps],
                boot_values[i].item(),
            )
            all_advantages.append(adv_i)
            all_returns.append(ret_i)
            for s in steps:
                self.buffer.add(
                    node_features=s["node_features"],
                    action=s["action"],
                    log_prob=s["log_prob"],
                    reward=s["reward"],
                    value=s["value"],
                    done=s["done"],
                )

        advantages = torch.cat(all_advantages)
        returns = torch.cat(all_returns)
        return advantages, returns, rollout_crossing_rates

    def update(self):
        advantages, returns, rollout_crossing_rates = self.collect_rollouts()
        advantages = (advantages - advantages.mean()) / (advantages.std() +
                                                         1e-8)
        ret_mean = returns.mean()
        ret_std = returns.std() + 1e-8

        total_loss = total_pg = total_vl = total_ent = 0
        indices = np.arange(len(self.buffer))

        # Pre-pad node_features once for all epochs (avoids repeated Python padding loops)
        padded_all, mask_all = self.buffer.get_padded_features()

        for _ in range(self.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, len(self.buffer), self.batch_size):
                end = min(start + self.batch_size, len(self.buffer))
                bi = indices[start:end]

                batch_actions = self.buffer.actions[
                    bi]  # [B, 2], already on device

                batch_log_probs, batch_entropies, batch_values = \
                    self.policy.evaluate_action_prepadded(padded_all[bi], mask_all[bi], batch_actions)

                old_lp = self.buffer.log_probs[bi]  # [B], already on device
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
            "rollout_crossing_rates": rollout_crossing_rates,
        }

    def train(self):
        total_timesteps = self.args.total_timesteps
        log_interval = self.args.log_interval
        save_path = self.args.save_path

        n_updates = total_timesteps // self.n_steps

        print(f"Starting Sequential PPO training: {self.args.name}")
        print(f"Total timesteps: {total_timesteps}, Updates: {n_updates}")
        if self.start_update > 0:
            print(f"Resuming from update: {self.start_update}")
        print(f"Device: {self.device}")
        print("=" * 60)

        best_avg = float('inf')

        for update in tqdm(
                range(self.start_update + 1, n_updates + 1),
                desc="Training",
                initial=self.start_update,
                total=n_updates,
        ):
            stats = self.update()

            if update % log_interval == 0:
                rates = stats["rollout_crossing_rates"]
                avg_rate = np.mean(rates) if rates else float("nan")
                marker = " *" if (not np.isnan(avg_rate)
                                  and avg_rate < best_avg) else ""
                if not np.isnan(avg_rate) and avg_rate < best_avg:
                    best_avg = avg_rate
                log_dict = {
                    "global_step": update,
                    "crossing_rate": avg_rate,
                    "n_episodes": len(rates),
                    "loss": stats["loss"],
                    "pg_loss": stats["pg_loss"],
                    "value_loss": stats["value_loss"],
                    "entropy": stats["entropy"],
                }
                self.logger.log(log_dict)
                print(f"\nUpdate {update}/{n_updates}{marker}")
                print(
                    f"  Crossing Rate (per edge): {avg_rate:.4f}  [{len(rates)} eps]"
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
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        def smooth(data, window=50):
            return np.convolve(data, np.ones(window) / window, mode='valid') \
                if len(data) >= window else data

        if self.history["episode_crossings"]:
            ax = axes[0]
            d = self.history["episode_crossings"]
            ax.plot(d, alpha=0.2, color='blue')
            if len(d) >= 50:
                ax.plot(smooth(d), color='blue', linewidth=2)
            ax.set_xlabel("Episode")
            ax.set_ylabel("Total Crossings")
            ax.set_title("Total Crossings per Episode")
            ax.grid(True)

        if self.history["losses"]:
            ax = axes[1]
            ax.plot(self.history["losses"])
            ax.set_xlabel("Update")
            ax.set_ylabel("Loss")
            ax.set_title("Training Loss")
            ax.grid(True)

        plt.tight_layout()
        plt.savefig(Path(save_path) / "training_curves.png", dpi=150)
        plt.close()
