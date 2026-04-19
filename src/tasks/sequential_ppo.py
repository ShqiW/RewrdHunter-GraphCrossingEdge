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
from src.envs.refinement import RefinementEnvConfig
from src.models.transformer_policy import TransformerConfig
from src.models.continuous_gnn import ContinuousGNNConfig
from src.models.node_select_gnn import NodeSelectGNNConfig


@dataclass
class SequentialPPOArgs(BaseArgs):
    """
    Arguments for Sequential PPO training.

    CLI usage:
        --env.structure_weight 0.3
        --model.num_encoder_layers 3
    """
    name: str = "sequential_ppo"

    env: SequentialEnvConfig = field(default_factory=SequentialEnvConfig)
    model: TransformerConfig = field(default_factory=TransformerConfig)


@dataclass
class SequentialRefinementArgs(BaseArgs):
    """
    Arguments for Sequential Refinement training (GAT policy).

    CLI usage:
        --env.structure_weight 0.3
        --model.hidden_dim 128
    """
    name: str = "sequential_refinement"

    env: RefinementEnvConfig = field(default_factory=RefinementEnvConfig)
    model: ContinuousGNNConfig = field(
        default_factory=lambda: ContinuousGNNConfig(node_input_dim=5)
    )


@dataclass
class NodeSelectRefinementArgs(BaseArgs):
    """
    Node-selecting refinement: policy learns which node to move + displacement.

    CLI usage:
        --env.structure_weight 0.3
        --model.crossing_score_alpha 2.0
    """
    name: str = "node_select_refinement"

    env: RefinementEnvConfig = field(
        default_factory=lambda: RefinementEnvConfig(node_select_mode=True)
    )
    model: NodeSelectGNNConfig = field(
        default_factory=lambda: NodeSelectGNNConfig(node_input_dim=5)
    )
