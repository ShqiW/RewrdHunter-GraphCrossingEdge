"""
Standalone script: visualize before/after layouts for top-N test graphs.

Usage:
  # 默认参数
  poetry run python scripts/23_visualize_layouts.py

  # 指定不同的 checkpoint 和 data
  poetry run python scripts/23_visualize_layouts.py \
    --checkpoint checkpoints/rome_multi/checkpoint_1400.pt \
    --data_root data \
    --save_path results \
    --n 6
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from src.tasks.discrete_ppo import DiscretePPOArgs
from src.envs.discrete import DiscreteEnvConfig
from src.models.gnn import GNNConfig
from src.tasks.base import BasePPOConfig, BaseGraphConfig
from src.plot import plot_layouts


def main():
    parser = argparse.ArgumentParser(
        description="Visualize before/after PPO graph layouts")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to checkpoint .pt file (overrides save_path lookup)")
    parser.add_argument(
        "--save_path",
        default="checkpoints/rome_multi",
        help="Checkpoint directory (used to find final_model.pt or latest)")
    parser.add_argument("--data_root",
                        default="data",
                        help="Root directory for RomeDataset")
    parser.add_argument("--seed", type=int, default=42)
    cli = parser.parse_args()

    # Build args from defaults (model/env config will be loaded from checkpoint)
    args = DiscretePPOArgs(
        name="discrete_ppo",
        save_path=cli.save_path,
        graph=BaseGraphConfig(
            use_dataset=True,
            data_root=cli.data_root,
            data_split="test",
        ),
    )

    plot_layouts(args, checkpoint_path=cli.checkpoint)


if __name__ == "__main__":
    main()
