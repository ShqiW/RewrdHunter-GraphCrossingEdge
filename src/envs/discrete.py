"""
Discrete action space graph layout environment.

Action space: Discrete(num_nodes * 8)
Action encoding: action = node_id * 8 + direction_id
8 directions: ↑ ↗ → ↘ ↓ ↙ ← ↖

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

# 8 direction unit vectors (normalised)
DIRECTIONS = np.array(
    [[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]],
    dtype=np.float32,
)
DIRECTIONS /= np.linalg.norm(DIRECTIONS, axis=1, keepdims=True)
NUM_DIRECTIONS = 8


@dataclass
class DiscreteEnvConfig(BaseEnvConfig):
    """Discrete environment configuration"""
    crossing_weight: float = 1.0
    structure_weight: float = 0.3
    structure_method: str = "softmax"
    softmax_tau: Optional[float] = 1
    use_potential_shaping: bool = False


class DiscreteGraphEnv(GATGraphEnv):
    """
    Discrete action space graph layout environment.

    Action decoding: action = node_id * 8 + direction_id
    """

    def __init__(
        self,
        graph_data: GraphData,
        device,
        config: DiscreteEnvConfig,
    ) -> None:
        super().__init__(graph_data, device, config)
        self.action_space = spaces.Discrete(self.num_nodes * NUM_DIRECTIONS)
        self.observation_space = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(self.num_nodes, 3),
                                            dtype=np.float32)
        self.reset()

    def step(self, action: int):
        node_id = action // NUM_DIRECTIONS
        direction_id = action % NUM_DIRECTIONS
        delta = DIRECTIONS[direction_id] * self.move_scale

        old_pos = self.gls.positions[node_id].copy()
        old_potential = self.current_potential
        self.gls.update_position(
            node_id, np.clip(self.gls.positions[node_id] + delta, -1.0, 1.0))
        displacement = float(np.linalg.norm(self.gls.positions[node_id] - old_pos))

        new_crossings = self._compute_crossings(self.coords)
        new_structure = self._compute_structure(self.coords)
        new_potential = -(new_crossings +
                          self.structure_weight * new_structure)

        crossing_reward = (self.current_crossings -
                           new_crossings) / self.max_crossings
        structure_reward = self.current_structure - new_structure

        if self.use_potential_shaping:
            reward = (self.crossing_weight * crossing_reward +
                      self.structure_weight * structure_reward + 0.5 *
                      (0.99 * new_potential - old_potential))
        else:
            reward = (self.crossing_weight * crossing_reward +
                      self.structure_weight * structure_reward)

        self.current_crossings = new_crossings
        self.current_structure = new_structure
        self.current_potential = new_potential
        self.steps += 1

        if new_crossings < self.best_crossings:
            self.best_crossings = new_crossings
            self.no_improve_steps = 0
        else:
            self.no_improve_steps += 1

        terminated = False
        static = (self.config.min_effective_action > 0
                  and displacement < self.config.min_effective_action)
        truncated = self.steps >= self.config.max_steps or static
        if self.current_crossings == 0:
            terminated = True
            reward += 10.0

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
            "displacement": displacement,
            "static_truncation": static,
            "action_node": node_id,
            "action_direction": direction_id,
        }
        return self._get_node_features(
            self.coords), reward, terminated, truncated, info
