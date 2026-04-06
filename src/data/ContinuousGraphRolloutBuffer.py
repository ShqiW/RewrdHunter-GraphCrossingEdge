import copy

import numpy as np
import torch
from torch_geometric.data import Data, Batch  # type: ignore[import]

from src.data.BaseRolloutBuffer import BaseRolloutBuffer


class ContinuousGraphRolloutBuffer(BaseRolloutBuffer):
    """
    Rollout buffer for continuous graph action spaces (ContinuousGraphEnv).

    Actions have shape (num_nodes, 2) which varies per graph, so they are
    stored as a list of tensors (same pattern as node_features in discrete buffer).
    Graph topology (edge_index, edge_attr) is also stored per step.
    """

    def _alloc(self):
        super()._alloc()
        self.node_features: list = [None] * self.n_steps
        self.edge_indices:  list = [None] * self.n_steps
        self.edge_attrs:    list = [None] * self.n_steps
        self.actions:       list = [None] * self.n_steps

    def add(self, node_features, edge_index, edge_attr, action,
            log_prob, reward, value, done):
        i = self.ptr
        self.node_features[i] = node_features
        self.edge_indices[i]  = edge_index
        self.edge_attrs[i]    = edge_attr
        # Store as float32 tensor on CPU to avoid GPU memory fragmentation
        self.actions[i]       = torch.as_tensor(action, dtype=torch.float32)
        self.log_probs[i]     = log_prob
        self.rewards[i]       = reward
        self.values[i]        = value
        self.dones[i]         = done
        self.ptr += 1

    # ------------------------------------------------------------------
    # PPO interface
    # ------------------------------------------------------------------

    def prepare_update(self):
        """No-op: variable-size actions don't need pre-padding."""
        pass

    def get_batch_data(self, indices):
        """
        Build a PyG Batch of graph snapshots + list of action tensors.

        Returns:
            batched:  PyG Batch (x, edge_index, edge_attr)
            actions:  list[Tensor], each [num_nodes_i, 2]
        """
        data_list = [
            Data(
                x=torch.tensor(self.node_features[i], dtype=torch.float32),
                edge_index=self.edge_indices[i],
                edge_attr=self.edge_attrs[i],
            )
            for i in indices
        ]
        batched = Batch.from_data_list(data_list).to(self.device)
        actions = [self.actions[i] for i in indices]
        return batched, actions

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def collect(self, policy, env_base, n_steps, n_envs, device, dataset,
                args, compute_gae_fn):
        """
        Collect n_steps of experience from n_envs parallel environments.

        Returns:
            advantages:  [≤n_steps] tensor
            returns:     [≤n_steps] tensor
            collect_log: dict with episode metrics and sampled_graphs list
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

        self._ensure_envs(
            n_envs, _make_env,
            list(range(n_envs)) if self._envs is None else [],
        )
        envs     = self._envs
        obs_list = self._obs_list

        env_steps = [[] for _ in range(n_envs)]
        episode_crossings, episode_improvements, episode_improvement_pcts = [], [], []
        episode_best_crossings, episode_steps, episode_static = [], [], []
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
                step_results = [envs[i].step(actions[i]) for i in range(n_envs)]

            for i, (next_obs, reward, terminated, truncated, info) in enumerate(step_results):
                done = terminated or truncated
                gd = envs[i].get_graph_data()
                env_steps[i].append(dict(
                    obs=obs_list[i].copy(),
                    edge_index=gd["edge_index"].clone(),
                    edge_attr=gd["edge_attr"].clone(),
                    action=actions[i],          # np.ndarray (num_nodes, 2)
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
                    episode_best_crossings.append(info.get("best_crossings", info.get("crossings", 0)))
                    episode_steps.append(info.get("steps", 0))
                    episode_static.append(float(info.get("static_truncation", False)))
                    self._ensure_envs(n_envs, _make_env, [i])
                else:
                    obs_list[i] = next_obs

        # Bootstrap
        batch_input = envs[0].make_batch_input(envs, obs_list, device)
        with torch.no_grad():
            _, _, boot_values = policy.get_action_batched(batch_input)

        all_advantages, all_returns = [], []
        for i in range(n_envs):
            steps = env_steps[i]
            if not steps:
                continue
            adv_i, ret_i = compute_gae_fn(
                [s["reward"] for s in steps],
                [s["value"]  for s in steps],
                [s["done"]   for s in steps],
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
        returns    = torch.cat(all_returns)

        _n = len(episode_improvements)
        collect_log = {
            "crossing_reduction":     np.mean(episode_improvements)       if _n else float("nan"),
            "crossing_reduction_pct": np.mean(episode_improvement_pcts)   if _n else float("nan"),
            "final_crossings":        np.mean(episode_crossings)           if _n else float("nan"),
            "best_crossings":         np.mean(episode_best_crossings)      if _n else float("nan"),
            "avg_steps":              np.mean(episode_steps)               if _n else float("nan"),
            "static_truncation_rate": np.mean(episode_static)             if _n else float("nan"),
            "n_episodes":             _n,
            "sampled_graphs":         sampled_graphs,
        }
        return advantages, returns, collect_log
