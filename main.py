"""
Unified entry point for RewardHunter.

Usage:
    poetry run python main.py --name discrete_ppo --total-timesteps 50000
    poetry run python main.py --name discrete_ppo --env.max-steps 200 --model.hidden-dim 256
"""
import tyro
from controller.cli.utils import auto_extract_args

from src.tasks.base import BaseArgs


def main():
    # Stage 1: Parse name using BaseArgs
    base_args, unknown = auto_extract_args(BaseArgs)

    # Stage 2: Match name and parse with specific Args class
    match base_args.name:
        case "discrete_ppo":
            from src.tasks.discrete_ppo import DiscretePPOArgs
            args = tyro.cli(DiscretePPOArgs)

        case "base":
            # For testing, allow running with base args
            args = tyro.cli(BaseArgs)

        case _:
            raise NotImplementedError(
                f"Task '{base_args.name}' not implemented. "
                f"Available tasks: discrete_ppo"
            )

    # Stage 3: Train
    from src.train import train
    policy, env, history = train(args)

    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Final model saved to: {args.save_path}")


if __name__ == "__main__":
    main()
