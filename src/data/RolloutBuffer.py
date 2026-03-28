import copy

import numpy as np
import torch
from torch_geometric.data import Data, Batch

from src.data.BaseRolloutBuffer import BaseRolloutBuffer


class RolloutBuffer(BaseRolloutBuffer):
    """Rollout buffer for discrete action spaces (graph-move PPO)."""

    def _alloc(self):
        super()._alloc()
        # Variable-size graph data: pre-sized lists avoid repeated append/resize
        self.node_features = [None] * self.n_steps
        self.edge_indices  = [None] * self.n_steps
        self.edge_attrs    = [None] * self.n_steps
        # Per-step scalar action (int index into flattened action space)
        self.actions = torch.zeros(self.n_steps, dtype=torch.long, device=self.device)
        # Persistent envs across rollouts (managed by BaseRolloutBuffer)

    def add(self, node_features, edge_index, edge_attr, action, log_prob,
            reward, value, done):
        i = self.ptr
        self.node_features[i] = node_features
        self.edge_indices[i]  = edge_index
        self.edge_attrs[i]    = edge_attr
        self.actions[i]   = action
        self.log_probs[i] = log_prob
        self.rewards[i]   = reward
        self.values[i]    = value
        self.dones[i]     = done
        self.ptr += 1

    # ------------------------------------------------------------------
    # PPO interface
    # ------------------------------------------------------------------

    def prepare_update(self):
        """No-op for discrete buffer; satisfies shared interface."""
        pass

    def get_batch_data(self, indices):
        """
        Build a PyG Batch from stored graph snapshots and return raw data.

        Returns:
            batched:       PyG Batch of graph snapshots
            batch_actions: [B] long tensor of actions
        """
        data_list = [
            Data(
                x=torch.tensor(self.node_features[i], dtype=torch.float32),
                edge_index=self.edge_indices[i],
                edge_attr=self.edge_attrs[i],
            ) for i in indices
        ]
        batched = Batch.from_data_list(data_list).to(self.device)
        return batched, self.actions[indices]

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def collect(self, policy, env_base, n_steps, n_envs, device, dataset, args,
                compute_gae_fn):
        """
        Collect n_steps of experience from n_envs parallel environments.

        Returns:
            advantages:   [≤n_steps] tensor
            returns:      [≤n_steps] tensor
            collect_log:  dict with episode metrics and sampled_graphs list
        """
        from src.trainer.utils import create_env

        self.reset()

        sampled_graphs = []

        def _make_env():
            if dataset is not None:
                gd = dataset.sample()
                sampled_graphs.append((gd.graph_name, gd.num_nodes))
                return create_env(args, device=device, graph_data=gd)
            return copy.deepcopy(env_base)

        self._ensure_envs(n_envs, _make_env, list(range(n_envs)) if self._envs is None else [])
        envs = self._envs
        obs_list = self._obs_list

        env_steps = [[] for _ in range(n_envs)]
        episode_crossings, episode_improvements, episode_improvement_pcts = [], [], []
        n_ticks = n_steps // n_envs

        executor = self._get_executor(n_envs)
        for _ in range(n_ticks):
            batch_input = envs[0].make_batch_input(envs, obs_list, device)
            with torch.no_grad():
                actions, log_probs, values = policy.get_action_batched(batch_input)

            if executor is not None:
                futures = [
                    executor.submit(envs[i].step, actions[i])
                    for i in range(n_envs)
                ]
                step_results = [f.result() for f in futures]
            else:
                step_results = [
                    envs[i].step(actions[i]) for i in range(n_envs)
                ]

            for i, (next_obs, reward, terminated, truncated,
                    info) in enumerate(step_results):
                done = terminated or truncated
                gd = envs[i].get_graph_data()
                env_steps[i].append(
                    dict(
                        obs=obs_list[i].copy(),
                        edge_index=gd["edge_index"].clone(),
                        edge_attr=gd["edge_attr"].clone(),
                        action=actions[i],
                        log_prob=log_probs[i].item(),
                        reward=reward,
                        value=values[i].item(),
                        done=done,
                    ))

                if done:
                    improvement = info.get("improvement", 0)
                    initial = info.get("initial_crossings", 0)
                    episode_crossings.append(info.get("crossings", 0))
                    episode_improvements.append(improvement)
                    episode_improvement_pcts.append(
                        improvement / initial * 100 if initial > 0 else 0.0)
                    self._ensure_envs(n_envs, _make_env, [i])
                else:
                    obs_list[i] = next_obs

        # Bootstrap: value estimates for last observation of each env
        batch_input = envs[0].make_batch_input(envs, obs_list, device)
        with torch.no_grad():
            _, _, boot_values = policy.get_action_batched(batch_input)

        # Per-env GAE → flatten into buffer
        all_advantages, all_returns = [], []
        for i in range(n_envs):
            steps = env_steps[i]
            if not steps:
                continue
            adv_i, ret_i = compute_gae_fn(
                [s["reward"] for s in steps],
                [s["value"] for s in steps],
                [s["done"] for s in steps],
                boot_values[i].item(),
            )
            all_advantages.append(adv_i)
            all_returns.append(ret_i)
            for s in steps:
                self.add(
                    node_features=s["obs"],
                    edge_index=s["edge_index"],
                    edge_attr=s["edge_attr"],
                    action=s["action"],
                    log_prob=s["log_prob"],
                    reward=s["reward"],
                    value=s["value"],
                    done=s["done"],
                )

        advantages = torch.cat(all_advantages)
        returns = torch.cat(all_returns)

        avg = np.mean(episode_improvements) if episode_improvements else float("nan")
        avg_pct = np.mean(episode_improvement_pcts) if episode_improvement_pcts else float("nan")
        collect_log = {
            "crossing_reduction": avg,
            "crossing_reduction_pct": avg_pct,
            "n_episodes": len(episode_improvements),
            "sampled_graphs": sampled_graphs,
        }
        return advantages, returns, collect_log
