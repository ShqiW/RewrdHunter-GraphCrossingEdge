"""
Base Args for all tasks.
Inherits from investigation's ExampleArgs.
"""
from dataclasses import dataclass, field
from typing import Optional
from investigation.doeargs.args import ExampleArgs


@dataclass
class BaseEnvConfig:
    """Base environment configuration"""
    type: str = "base"
    move_scale: float = 0.05
    initial_layout: str = "random"  # random, neato, sfdp, spring


@dataclass
class BaseModelConfig:
    """Base model configuration"""
    type: str = "base"
    hidden_dim: int = 128


@dataclass
class BaseRewardConfig:
    """Base reward configuration"""
    crossing_weight: float = 1.0
    use_potential_shaping: bool = True


@dataclass
class BasePPOConfig:
    """PPO algorithm configuration"""
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_steps: int = 128
    n_epochs: int = 4
    batch_size: int = 64
    n_envs: int = 1


@dataclass
class BaseGraphConfig:
    """Graph configuration"""
    # Single graph mode
    graph_path: Optional[str] = ""
    num_nodes: int = 15
    edge_prob: float = 0.15

    # Dataset mode
    use_dataset: bool = False
    data_root: str = "data"
    data_split: str = "train"  # train or test


@dataclass
class BaseArgs(ExampleArgs):
    """
    Base arguments for all tasks.

    Nested dataclasses will be flattened by investigation:
    - model.hidden_dim -> --model.hidden_dim
    """
    name: str = "base"

    # Nested configs
    env: BaseEnvConfig = field(default_factory=BaseEnvConfig)
    model: BaseModelConfig = field(default_factory=BaseModelConfig)
    reward: BaseRewardConfig = field(default_factory=BaseRewardConfig)
    ppo: BasePPOConfig = field(default_factory=BasePPOConfig)
    graph: BaseGraphConfig = field(default_factory=BaseGraphConfig)
    job_name: str = ""

    # Training
    total_timesteps: int = 50000
    save_path: str = "checkpoints"
    log_interval: int = 10

    if_train: bool = False
    if_plot: bool = False
    if_evaluate: bool = False
    baseline_csv: str = ""  # path to all_baselines.csv for comparison
