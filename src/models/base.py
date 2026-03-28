"""
Abstract base class for all policy networks.

Provides a unified interface for actor-critic policies used in PPO training,
enabling IDE type support and enforcing the common contract between
discrete (GNN) and continuous (Transformer) policy implementations.
"""
from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn as nn


class BasePolicy(nn.Module, ABC):
    """
    Abstract base class for actor-critic policies.

    Subclasses must implement:
        forward()                 — compute logits/mu/sigma + value
        get_action()              — single-step rollout inference
        evaluate_action_batched() — batch evaluation for PPO update

    evaluate_action_batched always returns (log_probs, entropies, values).
    Its positional arguments match whatever get_batch_data() returns from the
    corresponding RolloutBuffer, so train() can call it uniformly as:
        model.evaluate_action_batched(*buffer.get_batch_data(bi))
    """

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Full forward pass.

        Returns policy outputs and value estimate.
        Concrete signature varies by subclass (graph data vs. feature lists).
        """
        ...

    @abstractmethod
    def get_action(
        self,
        *args,
        deterministic: bool = False,
        **kwargs,
    ) -> Tuple:
        """
        Sample (or greedily select) one action for a single environment step.

        Returns:
            action:   sampled action (int for discrete, np.ndarray for continuous)
            log_prob: log probability of the action (torch scalar)
            value:    critic estimate (torch scalar)
        """
        ...

    @abstractmethod
    def get_action_batched(
        self,
        batch_input,
        deterministic: bool = False,
    ) -> Tuple:
        """
        Sample actions for a batch of environments in one forward pass.

        batch_input format is env-specific:
          DiscreteGraphEnv   → PyG Batch  (x, edge_index, edge_attr, batch, ptr)
          SequentialGraphEnv → List[Tensor]  (one [ni, 3] tensor per env)

        Returns:
            actions:   list of per-env actions (int for discrete, ndarray for continuous)
            log_probs: [B] tensor
            values:    [B] tensor
        """
        ...

    @abstractmethod
    def evaluate_action_batched(
        self,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate a batch of (state, action) pairs for PPO update.

        Returns:
            log_probs:  [B] — log probability of each action
            entropies:  [B] — entropy of each distribution
            values:     [B] — critic estimate for each state
        """
        ...
