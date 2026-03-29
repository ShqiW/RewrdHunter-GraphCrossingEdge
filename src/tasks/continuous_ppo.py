"""
Continuous PPO Args.

Uses:
- ContinuousGraphEnv with Softmax Ranking reward
- ContinuousGNNPolicy (Gaussian actor-critic)
- PPO algorithm
"""
from dataclasses import dataclass, field

from src.tasks.base import BaseArgs
from src.envs.continuous import ContinuousEnvConfig
from src.models.continuous_gnn import ContinuousGNNConfig


@dataclass
class ContinuousPPOArgs(BaseArgs):
    """Arguments for Continuous PPO training."""
    name: str = "continuous_ppo"

    env: ContinuousEnvConfig = field(default_factory=ContinuousEnvConfig)
    model: ContinuousGNNConfig = field(default_factory=ContinuousGNNConfig)
