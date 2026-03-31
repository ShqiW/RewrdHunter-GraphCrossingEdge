"""
Unified entry point for RewardHunter.

Usage:
    poetry run python main.py --name discrete_ppo --total-timesteps 50000
    poetry run python main.py --name discrete_ppo --env.max-steps 200 --model.hidden-dim 256
"""

import torch
from investigation.cli.utils import auto_extract_args
from pathlib import Path

from src.tasks.base import BaseArgs
from src.data.rome import RomeDataset
from src.envs.base import BaseGraphEnv
from src.models.base import BasePolicy
from src.plot import plot_discrete, render_gif
from tqdm import tqdm
import numpy as np


def main():
    # Stage 1: Parse name using BaseArgs
    base_args, unknown = auto_extract_args(BaseArgs)

    # Stage 2: Match name and parse with specific Args class
    match base_args.name:
        case "discrete_ppo":
            from src.tasks.discrete_ppo import DiscretePPOArgs
            from src.models.gnn import DiscreteGNNPolicy as ModelClass
            from src.envs.discrete import DiscreteGraphEnv as ENVClass
            from src.data.RolloutBuffer import RolloutBuffer as BufferClass
            # args = tyro.cli(DiscretePPOArgs)
            args, _ = auto_extract_args(DiscretePPOArgs)

        case "sequential_ppo":
            from src.tasks.sequential_ppo import SequentialPPOArgs
            from src.models.transformer_policy import TransformerPlacementPolicy as ModelClass
            from src.envs.sequential import SequentialGraphEnv as ENVClass
            from src.data.ContinuousRolloutBuffer import ContinuousRolloutBuffer as BufferClass
            # args = tyro.cli(SequentialPPOArgs)
            args, _ = auto_extract_args(SequentialPPOArgs)
        case "sequential_refinement":
            from src.tasks.sequential_ppo import SequentialRefinementArgs
            from src.models.gat_refinement_policy import GATRefinementPolicy as ModelClass
            from src.envs.refinement import RefinementGraphEnv as ENVClass
            from src.data.GATRefinementRolloutBuffer import GATRefinementRolloutBuffer as BufferClass
            args, _ = auto_extract_args(SequentialRefinementArgs)
        case "continuous_ppo":
            from src.tasks.continuous_ppo import ContinuousPPOArgs
            from src.models.continuous_gnn import ContinuousGNNPolicy as ModelClass
            from src.envs.continuous import ContinuousGraphEnv as ENVClass
            from src.data.ContinuousGraphRolloutBuffer import ContinuousGraphRolloutBuffer as BufferClass
            args, _ = auto_extract_args(ContinuousPPOArgs)
        case _:
            raise NotImplementedError(
                f"Task '{base_args.name}' not implemented. "
                f"Available tasks: discrete_ppo, sequential_ppo, sequential_refinement, continuous_ppo"
            )

    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.num_gpus > 0 else "cpu")

    dataset = RomeDataset(
        root=args.graph.data_root,
        split=args.graph.data_split,
        # force_reload=True,
    )
    print(f"Dataset: {len(dataset)} graphs from {args.graph.data_split} split")

    graph_data = dataset.sample()
    print(
        f"Initial graph: {graph_data.graph_name} ({graph_data.num_nodes} nodes)"
    )

    env: BaseGraphEnv = ENVClass(graph_data=graph_data,
                                 device=device,
                                 config=args.env)
    model: BasePolicy = ModelClass(config=args.model).to(device)
    buffer = BufferClass(args.model.n_steps, device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.name}, params: {n_params:,}")

    if args.if_train:
        from src.train import train
        model = train(args, device, model, dataset, env, buffer)

        print("\n" + "=" * 60)
        print("Training completed!")
        print(f"Final model saved to: {args.save_path}")

    final_ckpt = Path(args.save_path) / "final_model.pt"
    if (not final_ckpt.exists()):
        final_ckpt = sorted(Path(args.save_path).glob("checkpoint_*.pt"))[-1]
    if args.if_evaluate or args.if_plot:
        ckpt = torch.load(final_ckpt, map_location=device)

        model.load_state_dict(ckpt["policy_state_dict"])
        model.to(device).eval()
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
            args=args,
            policy=model,
            output_dir=eval_dir,
            data_root=args.graph.data_root or None,
            split="test",
            baseline_csv=baseline_csv,
            device=device,
            collect_frames=args.if_plot,
        )
        if comparison_rows and baseline_csv:
            from src.plot import plot_comparison
            plot_comparison(
                comparison_csv=str(Path(eval_dir) / "comparison.csv"),
                output_dir=eval_dir,
            )

        if args.if_plot:
            results.sort(
                key=lambda r: r.get("improvement", r["best_xing"]),
                reverse=True,
            )
            subset = results[:100]

            out_dir = Path("visualization")
            out_dir.mkdir(parents=True, exist_ok=True)
            plot_discrete(subset, out_dir)
            all_xings = [r["best_xing"] for r in results]
            zero_pct = 100 * sum(1
                                 for x in all_xings if x == 0) / len(all_xings)

            print(f"\nSaved {len(subset)} figures to {out_dir}/")
            print(
                f"Full test  — mean: {np.mean(all_xings):.2f}, "
                f"median: {np.median(all_xings):.1f}, "
                f"zero: {sum(1 for x in all_xings if x==0)}/{len(all_xings)}, "
                f"zero-crossing={zero_pct:.1f}%")

            # ── GIF rendering ──────────────────────────────────────────────────────────
            gif_dir = out_dir / "gifs"
            gif_dir.mkdir(exist_ok=True)
            for r in tqdm(results, desc="Rendering GIFs"):
                frames_data = r.get("frames")  # type: ignore[assignment]
                if not frames_data:
                    continue
                graph_name: str = r.get(
                    "graph_name")  # type: ignore[assignment]
                gif_path = gif_dir / f"{Path(graph_name).stem}.gif"
                render_gif(
                    frames=frames_data,
                    graph=r["graph"],
                    output_path=str(gif_path),
                    fps=24,
                    crossings=r["crossings"],
                    rewards=r["rewards"],
                )  # type: ignore[arg-type]
            print(f"GIFs saved to {gif_dir}/")


if __name__ == "__main__":
    main()
