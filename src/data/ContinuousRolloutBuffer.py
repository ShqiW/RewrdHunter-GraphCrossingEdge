import copy

import numpy as np
import torch

from src.data.BaseRolloutBuffer import BaseRolloutBuffer


class ContinuousRolloutBuffer(BaseRolloutBuffer):
    """
    Rollout buffer for continuous action spaces (sequential placement).

    Pre-allocates fixed-size tensors for scalar fields; node_features remain
    a list because each step has a variable number of placed nodes.
    """

    def _alloc(self):
        super()._alloc()
        self.node_features = [None] * self.n_steps
        self.actions = torch.zeros(
            self.n_steps,
            2,
            dtype=torch.float32,
            device=self.device,
        )

    def add(self, node_features, action, log_prob, reward, value, done):
        i = self.ptr
        self.node_features[i] = node_features
        self.actions[i] = torch.as_tensor(action,
                                          dtype=torch.float32,
                                          device=self.device)
        self.log_probs[i] = log_prob
        self.rewards[i] = reward
        self.values[i] = value
        self.dones[i] = done
        self.ptr += 1

    def get_padded_features(self):
        """
        Pad all stored node_features into a single tensor (run once before PPO epochs).

        Returns:
            padded: [n, max_n, 3] — pre-padded features on device
            mask:   [n, max_n]    — True = padding position
        """
        n = self.ptr
        sizes = [self.node_features[i].shape[0] for i in range(n)]
        max_n = max(sizes) if sizes and max(sizes) > 0 else 1
        feat_dim = self.node_features[0].shape[-1] if n > 0 else 4
        padded = torch.zeros(
            n,
            max_n,
            feat_dim,
            dtype=torch.float32,
            device=self.device,
        )
        mask = torch.ones(
            n,
            max_n,
            dtype=torch.bool,
            device=self.device,
        )
        for i in range(n):
            sz = sizes[i]
            if sz > 0:
                padded[i, :sz] = self.node_features[i]
                mask[i, :sz] = False
        return padded, mask

    # ------------------------------------------------------------------
    # PPO interface
    # ------------------------------------------------------------------

    def prepare_update(self):
        """Pre-pad all variable-length node features once before PPO epochs."""
        self._padded, self._mask = self.get_padded_features()

    def get_batch_data(self, indices):
        """
        Return pre-padded tensors for a batch of stored (state, action) pairs.

        Returns:
            padded:   [B, max_n, 3] node feature tensor
            mask:     [B, max_n]    padding mask (True = padding)
            actions:  [B, 2]        action tensor
        """
        return self._padded[indices], self._mask[indices], self.actions[
            indices]

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def collect(
        self,
        policy,
        env_base,
        n_steps,
        n_envs,
        device,
        dataset,
        args,
        compute_gae_fn,
    ):
        """
        Collect n_steps of experience from n_envs parallel environments.

        Returns:
            advantages:    [≤n_steps] tensor
            returns:       [≤n_steps] tensor
            stats:         dict with crossing_rates and sampled_graphs
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
            n_envs,
            _make_env,
            list(range(n_envs)) if self._envs is None else [],
        )
        envs = self._envs
        env_obs = self._obs_list

        env_steps = [[] for _ in range(n_envs)]
        crossing_rates = []
        n_ticks = n_steps // n_envs

        executor = self._get_executor(n_envs)
        for _ in range(n_ticks):
            obs_tensors = [
                torch.tensor(obs, dtype=torch.float32, device=device)
                for obs in env_obs
            ]
            with torch.no_grad():
                actions, log_probs, values = policy.get_action_batched(
                    obs_tensors)

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
                    rate = total_xing / envs[i].num_edges \
                        if envs[i].num_edges > 0 else 0.0
                    crossing_rates.append(rate)
                    self._ensure_envs(n_envs, _make_env, [i])
                else:
                    env_obs[i] = next_obs

        # Bootstrap: one batched forward pass for all N last observations
        last_obs = [
            torch.tensor(obs, dtype=torch.float32, device=device)
            for obs in env_obs
        ]
        with torch.no_grad():
            _, _, boot_values = policy.forward(last_obs)

        # Per-env GAE, then flatten into buffer
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
                    node_features=s["node_features"],
                    action=s["action"],
                    log_prob=s["log_prob"],
                    reward=s["reward"],
                    value=s["value"],
                    done=s["done"],
                )

        avg_rate = np.mean(crossing_rates) if crossing_rates else float("nan")
        collect_log = {
            "crossing_rate": avg_rate,
            "n_episodes": len(crossing_rates),
            "sampled_graphs": sampled_graphs,
        }
        return torch.cat(all_advantages), torch.cat(all_returns), collect_log
