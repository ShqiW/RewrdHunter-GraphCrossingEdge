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
    move_scale: float = 0.05
    initial_layout: str = "neato"  # random, neato, sfdp, spring
    max_steps: int = 128
    # Static-action early termination: if the actual per-node displacement after
    # clipping falls below this threshold the state cannot change, so the episode
    # is truncated immediately.  Set to 0 to disable.
    min_effective_action: float = 1e-4
    # Soft crossing reward: use differentiable sigmoid approximation for the
    # crossing term in the reward (denser gradient signal).  Hard integer count
    # is still used for termination and evaluation.
    soft_crossing: bool = False
    soft_crossing_sharpness: float = 10.0


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
    hidden_dim: int = 128


@dataclass
class BaseGraphConfig:
    """Graph configuration"""
    # Single graph mode

    # Dataset mode
    data_root: str = "data"
    data_split: str = "train"  # train or test


@dataclass
class BaseArgs(ExampleArgs):
    """
    Base arguments for all tasks.

    Nested dataclasses will be flattened by investigation:
    """
    name: str = "base"

    # Nested configs
    env: BaseEnvConfig = field(default_factory=BaseEnvConfig)
    model: BasePPOConfig = field(default_factory=BasePPOConfig)
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
