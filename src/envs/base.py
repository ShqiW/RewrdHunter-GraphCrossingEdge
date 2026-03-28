"""
Abstract base class for graph layout environments.

Provides a unified interface for discrete (node-move) and sequential
(node-placement) environments used in PPO training.
"""
from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import torch
import gymnasium as gym


class BaseGraphEnv(gym.Env, ABC):
    """
    Abstract base class for graph layout gym environments.

    Subclasses must implement the gymnasium interface (reset / step) plus
    three rollout-support methods:

      get_graph_data()     — current graph tensors (env-specific dict)
      get_coords()         — current [num_nodes, 2] coordinates
      get_policy_input()   — single-env policy.get_action(*args) inputs
      make_batch_input()   — batched policy.get_action_batched(input) input

    The two policy-input methods decouple the buffer's rollout loop from
    env-specific data formats (PyG Batch for discrete, tensor list for
    sequential), so both buffer types can share the same collect() structure.

    Common attributes set by subclasses:
      num_nodes:   int
      num_edges:   int
      graph_name:  str | None
    """

    num_nodes: int
    num_edges: int
    graph_name: str | None

    # ── Graph state ────────────────────────────────────────────────────────────

    @abstractmethod
    def get_graph_data(self) -> Dict:
        """
        Return current graph state tensors for the policy network.

        DiscreteGraphEnv returns:   {edge_index, edge_attr, num_nodes, degree}
        SequentialGraphEnv returns: {node_features, edge_index, placed_nodes}
        """
        ...

    @abstractmethod
    def get_coords(self) -> np.ndarray:
        """Return current node coordinates as [num_nodes, 2] float32 array."""
        ...

    # ── Policy-input interface ─────────────────────────────────────────────────

    @abstractmethod
    def get_policy_input(self, obs: np.ndarray, device: torch.device) -> tuple:
        """
        Prepare args for policy.get_action(*args) from a single obs array.

        DiscreteGraphEnv:   returns (node_features, edge_index, edge_attr)
        SequentialGraphEnv: returns (node_features_tensor,)
        """
        ...

    @staticmethod
    @abstractmethod
    def make_batch_input(envs: List['BaseGraphEnv'], obs_list: list,
                         device: torch.device):
        """
        Prepare the single argument for policy.get_action_batched(batch_input)
        from a list of environment instances and their current observations.

        DiscreteGraphEnv:   returns a PyG Batch (x, edge_index, edge_attr)
        SequentialGraphEnv: returns a list of node-feature tensors
        """
        ...
