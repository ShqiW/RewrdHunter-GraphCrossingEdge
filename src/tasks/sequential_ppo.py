"""
Sequential PPO task.

Uses:
- SequentialGraphEnv: BFS-ordered node placement, continuous 2D actions
- TransformerPlacementPolicy: Transformer encoder + Gaussian actor-critic
- PPO with continuous actions
"""
from dataclasses import dataclass, field

from src.tasks.base import BaseArgs, BasePPOConfig
from src.envs.sequential import SequentialEnvConfig
from src.models.transformer_policy import TransformerConfig


@dataclass
class SequentialPPOArgs(BaseArgs):
    """
    Arguments for Sequential PPO training.

    CLI usage:
        --env.structure_weight 0.3
        --model.hidden_dim 128
        --model.num_encoder_layers 3
        --ppo.lr 3e-4
        --graph.use_dataset True
    """
    name: str = "sequential_ppo"

    env: SequentialEnvConfig = field(default_factory=SequentialEnvConfig)
    model: TransformerConfig = field(default_factory=TransformerConfig)
    ppo: BasePPOConfig = field(default_factory=BasePPOConfig)
