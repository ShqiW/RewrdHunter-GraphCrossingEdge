"""
Unified entry point for RewardHunter.

Usage:
    poetry run python main.py --name discrete_ppo --total-timesteps 50000
    poetry run python main.py --name discrete_ppo --env.max-steps 200 --model.hidden-dim 256
"""
import tyro
from investigation.cli.utils import auto_extract_args
from pathlib import Path

from src.tasks.base import BaseArgs


def main():
    # Stage 1: Parse name using BaseArgs
    base_args, unknown = auto_extract_args(BaseArgs)

    # Stage 2: Match name and parse with specific Args class
    match base_args.name:
        case "discrete_ppo":
            from src.tasks.discrete_ppo import DiscretePPOArgs
            args = tyro.cli(DiscretePPOArgs)

        case "sequential_ppo":
            from src.tasks.sequential_ppo import SequentialPPOArgs
            args = tyro.cli(SequentialPPOArgs)

        case "base":
            # For testing, allow running with base args
            args = tyro.cli(BaseArgs)

        case _:
            raise NotImplementedError(
                f"Task '{base_args.name}' not implemented. "
                f"Available tasks: discrete_ppo, sequential_ppo")

    if (args.if_train):
        # Stage 3: Train
        from src.train import train
        policy, env, history = train(args)

        print("\n" + "=" * 60)
        print("Training completed!")
        print(f"Final model saved to: {args.save_path}")

    if (args.if_plot):
        from src.plot import plot_layouts
        plot_layouts(args)

    if (args.if_evaluate):
        from src.evaluate import evaluate
        ckpt_dir = Path(args.save_path)
        ckpt_path = ckpt_dir / "final_model.pt"
        if not ckpt_path.exists():
            checkpoints = sorted(ckpt_dir.glob("checkpoint_*.pt"))
            if not checkpoints:
                raise FileNotFoundError(
                    f"No checkpoint found in {args.save_path}")
            ckpt_path = checkpoints[-1]
        eval_dir = "eval_results"
        baseline_csv = args.baseline_csv or None
        results, comparison_rows = evaluate(
            checkpoint_path=str(ckpt_path),
            output_dir=eval_dir,
            data_root=args.graph.data_root or None,
            split="test",
            baseline_csv=baseline_csv,
        )
        if comparison_rows and baseline_csv:
            from src.plot import plot_comparison
            plot_comparison(
                comparison_csv=str(Path(eval_dir) / "comparison.csv"),
                output_dir=eval_dir,
            )


if __name__ == "__main__":
    main()
