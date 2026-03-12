"""
Discrete PPO Args.

Uses:
- DiscreteGraphEnv with Softmax Ranking reward
- DiscreteGNNPolicy
- PPO algorithm
"""
from dataclasses import dataclass, field

from src.tasks.base import (BaseArgs, BasePPOConfig)
from src.envs.discrete import DiscreteEnvConfig
from src.models.gnn import GNNConfig


@dataclass
class DiscretePPOArgs(BaseArgs):
    """
    Arguments for Discrete PPO training.

    All nested configs can be set via CLI:
    --env.max_steps 200
    --model.hidden_dim 256
    --ppo.lr 1e-4
    """
    name: str = "discrete_ppo"

    # Nested configs with task-specific defaults
    env: DiscreteEnvConfig = field(default_factory=DiscreteEnvConfig)
    model: GNNConfig = field(default_factory=GNNConfig)
    ppo: BasePPOConfig = field(default_factory=BasePPOConfig)
