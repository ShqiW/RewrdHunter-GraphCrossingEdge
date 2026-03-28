"""
Abstract base class for rollout buffers.

Defines the unified interface for PPO rollout collection and batch access.
Subclasses implement env-specific storage and collection logic while sharing
pre-allocated scalar tensors and common methods.
"""
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

import torch


class BaseRolloutBuffer(ABC):

    def __init__(self, n_steps: int, device: torch.device):
        self.n_steps = n_steps
        self.device = device
        self._alloc()

    def _alloc(self):
        """Pre-allocate shared scalar tensors. Subclasses call super() then add their own."""
        self.log_probs = torch.zeros(self.n_steps,
                                     dtype=torch.float32,
                                     device=self.device)
        self.rewards = torch.zeros(self.n_steps,
                                   dtype=torch.float32,
                                   device=self.device)
        self.values = torch.zeros(self.n_steps,
                                  dtype=torch.float32,
                                  device=self.device)
        self.dones = torch.zeros(self.n_steps,
                                 dtype=torch.bool,
                                 device=self.device)
        self.ptr = 0
        self._envs = None
        self._obs_list = None
        self._executor = None

    def _get_executor(self, n_envs: int):
        """Return a persistent thread pool (created once, reused across updates)."""
        if n_envs > 1 and self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=n_envs)
        return self._executor

    def _ensure_envs(self, n_envs, make_env_fn, done_indices):
        """
        Create or replace env slots at done_indices with new instances.
        First call: pass list(range(n_envs)) to initialize all slots.
        Subsequent calls: pass [i] to replace a single done env.
        """
        if self._envs is None:
            self._envs = [None] * n_envs
            self._obs_list = [None] * n_envs
        for i in done_indices:
            self._envs[i] = make_env_fn()
            self._obs_list[i], _ = self._envs[i].reset()

    def reset(self):
        self.ptr = 0

    def __len__(self):
        return self.ptr

    def get_old_log_probs(self, indices):
        return self.log_probs[indices]

    @abstractmethod
    def add(self, *args, **kwargs):
        ...

    @abstractmethod
    def prepare_update(self):
        ...

    @abstractmethod
    def get_batch_data(self, indices):
        """Return raw data needed to call model.evaluate_action_batched()."""
        ...

    @abstractmethod
    def collect(self, policy, env_base, n_steps, n_envs, device, dataset, args,
                compute_gae_fn):
        """
        Collect n_steps of experience and compute GAE.

        Returns:
            advantages:   [≤n_steps] tensor
            returns:      [≤n_steps] tensor
            collect_log:  dict of loggable metrics from this rollout
        """
        ...
