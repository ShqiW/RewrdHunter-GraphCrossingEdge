"""
Rollout buffer for GATRefinementPolicy + RefinementGraphEnv.

Each step: obs [N, 5], edge_index [2, 2E], edge_attr [2E, 1], action [2].
Builds PyG Batches for efficient batched PPO updates.
"""
import copy

import numpy as np
import torch
from torch_geometric.data import Data, Batch

from src.data.BaseRolloutBuffer import BaseRolloutBuffer


class GATRefinementRolloutBuffer(BaseRolloutBuffer):

    def _alloc(self):
        super()._alloc()
        self.node_features: list = [None] * self.n_steps
        self.edge_indices:  list = [None] * self.n_steps
        self.edge_attrs:    list = [None] * self.n_steps
        self.actions:       list = [None] * self.n_steps  # each [2] np.ndarray

    def add(self, node_features, edge_index, edge_attr, action,
            log_prob, reward, value, done):
        i = self.ptr
        self.node_features[i] = node_features
        self.edge_indices[i]  = edge_index
        self.edge_attrs[i]    = edge_attr
        self.actions[i]       = torch.as_tensor(action, dtype=torch.float32)
        self.log_probs[i]     = log_prob
        self.rewards[i]       = reward
        self.values[i]        = value
        self.dones[i]         = done
        self.ptr += 1

    # ── PPO interface ─────────────────────────────────────────────────────────

    def prepare_update(self):
        n = self.ptr
        self._data_list = [
            Data(
                x=torch.tensor(self.node_features[i], dtype=torch.float32),
                edge_index=self.edge_indices[i],
                edge_attr=self.edge_attrs[i],
            )
            for i in range(n)
        ]
        self._actions_list = [
            torch.as_tensor(self.actions[i], dtype=torch.float32)
            for i in range(n)
        ]

    def get_batch_data(self, indices):
        sub_batch = Batch.from_data_list(
            [self._data_list[i] for i in indices]
        ).to(self.device)
        actions_t = torch.stack(
            [self._actions_list[i] for i in indices]
        ).to(self.device)
        return sub_batch, actions_t

    # ── Rollout collection ────────────────────────────────────────────────────

    def collect(self, policy, env_base, n_steps, n_envs, device, dataset,
                args, compute_gae_fn):
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
        episode_crossings, episode_steps, episode_static = [], [], []
        n_ticks = n_steps // n_envs

        for _ in range(n_ticks):
            # Build PyG batch from current obs + graph topology (before step)
            gd_list = [envs[i].get_graph_data() for i in range(n_envs)]
            data_list = [
                Data(
                    x=torch.tensor(obs_list[i], dtype=torch.float32, device=device),
                    edge_index=gd_list[i]["edge_index"].to(device),
                    edge_attr=gd_list[i]["edge_attr"].to(device),
                )
                for i in range(n_envs)
            ]
            pyg_batch = Batch.from_data_list(data_list)

            with torch.no_grad():
                actions, log_probs, values = policy.get_action_batched(pyg_batch)

            step_results = [envs[i].step(actions[i]) for i in range(n_envs)]

            for i, (next_obs, reward, terminated, truncated, info) in enumerate(step_results):
                done = terminated or truncated
                env_steps[i].append(dict(
                    obs=obs_list[i].copy(),
                    edge_index=gd_list[i]["edge_index"].clone(),
                    edge_attr=gd_list[i]["edge_attr"].clone(),
                    action=actions[i],
                    log_prob=log_probs[i].item(),
                    reward=reward,
                    value=values[i].item(),
                    done=done,
                ))
                if done:
                    episode_crossings.append(
                        info.get("total_crossings", info.get("crossings", 0)))
                    episode_steps.append(info.get("step", 0))
                    episode_static.append(float(info.get("static_truncation", False)))
                    self._ensure_envs(n_envs, _make_env, [i])
                else:
                    obs_list[i] = next_obs

        # Bootstrap values from current obs
        boot_gd_list = [envs[i].get_graph_data() for i in range(n_envs)]
        boot_data = [
            Data(
                x=torch.tensor(obs_list[i], dtype=torch.float32, device=device),
                edge_index=boot_gd_list[i]["edge_index"].to(device),
                edge_attr=boot_gd_list[i]["edge_attr"].to(device),
            )
            for i in range(n_envs)
        ]
        boot_batch = Batch.from_data_list(boot_data)
        with torch.no_grad():
            _, _, boot_values = policy.get_action_batched(boot_batch)

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

        _n = len(episode_crossings)
        avg_xing = np.mean(episode_crossings) if _n else float("nan")
        _num_edges = envs[0].num_edges if envs and envs[0] is not None else 1
        collect_log = {
            "final_crossings":        avg_xing,
            "crossing_rate":          avg_xing / _num_edges if _num_edges > 0 else float("nan"),
            "avg_steps":              np.mean(episode_steps)  if _n else float("nan"),
            "static_truncation_rate": np.mean(episode_static) if _n else float("nan"),
            "n_episodes":             _n,
            "sampled_graphs":         sampled_graphs,
        }
        return torch.cat(all_advantages), torch.cat(all_returns), collect_log
