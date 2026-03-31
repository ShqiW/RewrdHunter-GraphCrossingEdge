"""
Continuous action space graph layout environment.

Action space: Box(-move_scale, move_scale, shape=(num_nodes, 2))
Each step moves ALL nodes simultaneously by a (dx, dy) delta.

State Space (GAT features):
- Node features: [x, y, degree]
- Edge features: [edge_length]
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
from gymnasium import spaces

from src.data.data import GraphData
from src.envs.gat_env import GATGraphEnv
from src.tasks.base import BaseEnvConfig


@dataclass
class ContinuousEnvConfig(BaseEnvConfig):
    """Continuous environment configuration"""
    crossing_weight: float = 1.0
    structure_weight: float = 0.3
    structure_method: str = "softmax"
    softmax_tau: Optional[float] = 1
    use_potential_shaping: bool = False
    # Monte Carlo terminal reward: reward is 0 for intermediate steps and
    # (initial - final) / max_crossings at episode end.
    # When enabled, the recommended PPO gamma = terminal_reward_retention^(1/max_steps),
    # i.e. the terminal reward is discounted by `terminal_reward_retention` when
    # seen from the start of the episode.
    use_monte_carlo_reward: bool = False
    terminal_reward_retention: float = 0.5  # f: gamma^max_steps = f
    # Static-action early termination: if the max per-node L2 displacement after
    # clipping falls below this threshold the state cannot change, so the episode
    # is truncated immediately.  Set to 0 to disable.
    min_effective_action: float = 1e-4
    # Patience: truncate if crossings have not improved for this many steps.
    # Set to 0 to disable.
    patience: int = 50


class ContinuousGraphEnv(GATGraphEnv):
    """
    Continuous action space graph layout environment.

    Each step moves ALL nodes simultaneously by their (dx, dy) deltas.
    """

    def __init__(
        self,
        graph_data: GraphData,
        device,
        config: ContinuousEnvConfig,
    ) -> None:
        super().__init__(graph_data, device, config)
        self.config: ContinuousEnvConfig
        self.action_space = spaces.Box(
            low=-self.move_scale,
            high=self.move_scale,
            shape=(self.num_nodes, 2),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_nodes, 3),
            dtype=np.float32,
        )
        self.reset()

    def _compute_area_penalty(self, coords: np.ndarray) -> float:
        """
        Penalize near-1D layouts: std_x * std_y ≈ 0.
        Scale: O(num_edges) to match crossing-count magnitude.
        """
        area = float(coords[:, 0].std()) * float(coords[:, 1].std())
        target = 0.02
        if area >= target:
            return 0.0
        return self.num_edges * (target - area) / target

    def step(self, action: np.ndarray):
        """
        Args:
            action: (num_nodes, 2) deltas in [-move_scale, move_scale].
        """
        old_positions = self.gls.positions.copy()
        self.gls.batch_update(
            range(self.num_nodes),
            np.clip(self.gls.positions + action, -1.0, 1.0),
        )
        max_displacement = float(
            np.max(np.linalg.norm(self.gls.positions - old_positions, axis=1)))

        new_crossings = self._compute_crossings(self.coords)
        self.steps += 1

        if new_crossings < self.best_crossings:
            self.best_crossings = new_crossings
            self.no_improve_steps = 0
        else:
            self.no_improve_steps += 1

        terminated = new_crossings == 0
        static = (self.config.min_effective_action > 0
                  and max_displacement < self.config.min_effective_action)
        patience_exceeded = (self.config.patience > 0
                             and self.no_improve_steps >= self.config.patience)
        truncated = self.steps >= self.config.max_steps or static or patience_exceeded

        if self.config.use_monte_carlo_reward:
            # Reward is 0 for intermediate steps; terminal reward is the
            # normalised improvement over the full episode.
            if terminated or truncated:
                reward = float((self.initial_crossings - new_crossings) /
                               self.max_crossings * self.crossing_weight)
                if terminated:
                    reward += 10.0
            else:
                reward = 0.0
        else:
            old_potential = self.current_potential
            if self.soft_crossing:
                new_soft_crossings = self._compute_soft_crossings(self.coords)
                reward = ((self.current_soft_crossings - new_soft_crossings) /
                          self.max_crossings * self.crossing_weight)
                self.current_soft_crossings = new_soft_crossings
            else:
                reward = (self.current_crossings - new_crossings
                          ) / self.max_crossings * self.crossing_weight
            if self.structure_weight > 0 or self.use_potential_shaping:
                new_structure = self._compute_structure(self.coords)
                reward += (self.current_structure -
                           new_structure) * self.structure_weight
                self.current_structure = new_structure
                if self.use_potential_shaping:
                    new_potential = -(new_crossings +
                                      self.structure_weight * new_structure)
                    reward += 0.5 * (0.99 * new_potential - old_potential)
                    self.current_potential = new_potential
            if terminated:
                reward += 10.0
        self.current_crossings = new_crossings

        info = {
            "crossings": self.current_crossings,
            "structure_loss": self.current_structure,
            "best_crossings": self.best_crossings,
            "initial_crossings": self.initial_crossings,
            "initial_structure": self.initial_structure,
            "improvement": self.initial_crossings - self.current_crossings,
            "structure_improvement":
            self.initial_structure - self.current_structure,
            "steps": self.steps,
            "max_displacement": max_displacement,
            "static_truncation": static,
        }
        return self._get_node_features(
            self.coords), reward, terminated, truncated, info
