"""
Script to run RL training for graph layout optimization.
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.train import train_multi_graph, train_single_graph


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train RL model for graph layout optimization")
    parser.add_argument(
        "--mode",
        type=str,
        default="multi",
        choices=["single", "multi"],
        help="Training mode: 'single' for one graph, 'multi' for generalizable model"
    )
    parser.add_argument(
        "--graph",
        type=str,
        default=None,
        help="Path to single graph (for single mode)"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100000,
        help="Total training timesteps"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints"
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
        help="Hidden dimension for policy network"
    )
    parser.add_argument(
        "--move_scale",
        type=float,
        default=0.1,
        help="Scale factor for node movements"
    )
    args = parser.parse_args()

    print("=" * 50)
    print("Graph Layout Optimization with RL")
    print("=" * 50)
    print(f"Mode: {args.mode}")
    print(f"Device: {args.device}")
    print(f"Timesteps: {args.timesteps}")
    print(f"Save path: {args.save_path}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"Move scale: {args.move_scale}")
    print("=" * 50)

    if args.mode == "single":
        if args.graph is None:
            args.graph = str(PROJECT_ROOT / "data/rome/rome/grafo10000.38.graphml")
        print(f"Training on single graph: {args.graph}")
        train_single_graph(
            graph_path=args.graph,
            total_timesteps=args.timesteps,
            device=args.device,
            save_path=args.save_path,
        )
    else:
        print("Training generalizable model on training set...")
        train_multi_graph(
            graph_list_file=str(PROJECT_ROOT / "data/train_graph.txt"),
            data_dir=str(PROJECT_ROOT / "data"),
            total_timesteps=args.timesteps,
            device=args.device,
            save_path=args.save_path,
            hidden_dim=args.hidden_dim,
            move_scale=args.move_scale,
        )


if __name__ == "__main__":
    main()
