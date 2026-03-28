"""
Evaluate trained model on test set.

Usage:
    poetry run python src/evaluate.py \\
        --checkpoint .investigation/924863c/cases/Y6WG16I6/checkpoints/final_model.pt \\
        --output results/Y6WG16I6

Outputs:
    results.csv     — per-graph metrics
    summary.txt     — aggregate statistics
"""
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from src.envs.base import BaseGraphEnv
from src.envs.sequential import SequentialGraphEnv
from src.envs.discrete import DiscreteGraphEnv
from src.models.base import BasePolicy
from src.plot import build_nx_graph, count_hard_crossings
from src.tasks.base import BaseArgs
from typing import List, Dict
from src.envs.discrete import DiscreteEnvConfig, DiscreteGraphEnv
from src.envs.sequential import SequentialEnvConfig, SequentialGraphEnv
from src.envs.refinement import RefinementGraphEnv, RefinementEnvConfig
from src.tasks.sequential_ppo import SequentialPPOArgs, SequentialRefinementArgs
from src.tasks.discrete_ppo import DiscretePPOArgs
from src.data.data import GraphData
import pandas as pd
import networkx as nx
# ── helpers ────────────────────────────────────────────────────────────────────

# ── per-graph runners ──────────────────────────────────────────────────────────


def _run_sequential(
    policy: BasePolicy,
    graph_data: GraphData,
    device,
    args: SequentialPPOArgs | SequentialRefinementArgs,
    collect_frames: bool,
):
    import networkx as nx
    match args:
        case SequentialPPOArgs():
            env = SequentialGraphEnv(
                graph_data=graph_data,
                device=device,
                config=args.env,
            )
        case SequentialRefinementArgs():
            env = RefinementGraphEnv(
                graph_data=graph_data,
                device=device,
                config=args.env,
            )
    before_coords = graph_data.neato_coords.numpy()
    initial_xing = graph_data.neato_xing

    obs, _ = env.reset()
    frames: List[np.ndarray] = [env.get_coords().copy()] if collect_frames else []
    done = False
    while not done:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            action, _, _ = policy.get_action(obs_t, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if collect_frames:
            frames.append(env.get_coords().copy())
    coords = env.get_coords()
    # Recompute final crossings with XingLoss for consistency with discrete/plot
    G = nx.Graph()
    G.add_nodes_from(range(graph_data.num_nodes))
    edges = graph_data.edge_index.T[:graph_data.edge_index.shape[1] //
                                    2].tolist()
    G.add_edges_from(edges)

    best_xing = count_hard_crossings(G, coords, device)
    return {
        "best_xing": best_xing,
        "coords": coords,
        "before_coords": before_coords,
        "initial_xing": initial_xing,
        "improvement": initial_xing - best_xing,
        "frames": frames,
    }


def _run_discrete(
    policy: BasePolicy,
    graph_data: GraphData,
    device,
    args: DiscretePPOArgs,
    collect_frames: bool,
):

    env: BaseGraphEnv = DiscreteGraphEnv(
        graph_data=graph_data,
        device=device,
        config=args.env,
    )
    obs, info = env.reset()
    # before_coords = env.get_coords()
    before_coords = graph_data.neato_coords.numpy()
    graph = build_nx_graph(graph_data)
    # Use precomputed neato_xing to align with all_baselines.csv
    initial_xing = graph_data.neato_xing
    best_xing = initial_xing
    best_coords = before_coords.copy()
    frames: List[np.ndarray] = [env.get_coords().copy()] if collect_frames else []
    done = False
    # done = True
    while not done:
        node_features = torch.tensor(obs, dtype=torch.float32, device=device)
        gd = env.get_graph_data()
        with torch.no_grad():
            action, _, _ = policy.get_action(
                node_features,
                gd["edge_index"].to(device),
                gd["edge_attr"].to(device),
                deterministic=True,
            )
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if collect_frames:
            frames.append(env.get_coords().copy())

    # Compare final state against initial using hard crossings.
    # Soft-crossing tracking was previously used here, but with neato
    # initialization the initial soft crossing is already ≈ 0, so the
    # condition was never triggered and best_coords always equalled
    # before_coords, producing identical before/after plots.
    final_coords = env.get_coords()
    final_xing = count_hard_crossings(graph, final_coords, device)
    if final_xing <= best_xing:
        best_xing = final_xing
        best_coords = final_coords

    return {
        "coords": best_coords,
        "before_coords": before_coords,
        "initial_xing": initial_xing,
        "best_xing": best_xing,
        "improvement": initial_xing - best_xing,
        "frames": frames,
    }


# ── GIF rendering ─────────────────────────────────────────────────────────────


def render_gif(
    frames: List[np.ndarray],
    graph: nx.Graph,
    output_path: str,
    fps: int = 10,
    figsize: tuple = (5, 5),
) -> None:
    """
    Render a list of coordinate snapshots as an animated GIF.

    Args:
        frames:      List of [N, 2] float32 coord arrays (one per step).
        graph:       NetworkX graph (for drawing edges).
        output_path: Where to write the .gif file.
        fps:         Frames per second.
        figsize:     Matplotlib figure size.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    edges = list(graph.edges())
    fig, ax = plt.subplots(figsize=figsize)

    def _draw(coords: np.ndarray):
        ax.clear()
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect("equal")
        ax.axis("off")
        for u, v in edges:
            ax.plot(
                [coords[u, 0], coords[v, 0]],
                [coords[u, 1], coords[v, 1]],
                color="steelblue", linewidth=0.8, alpha=0.7,
            )
        ax.scatter(coords[:, 0], coords[:, 1], s=20, color="tomato", zorder=3)

    def _update(frame_idx):
        _draw(frames[frame_idx])
        ax.set_title(f"step {frame_idx}/{len(frames)-1}", fontsize=8)

    ani = animation.FuncAnimation(
        fig, _update, frames=len(frames), interval=1000 // fps, repeat=False
    )
    ani.save(output_path, writer="pillow", fps=fps)
    plt.close(fig)


# ── main evaluate ──────────────────────────────────────────────────────────────


def _compute_ratio(our: int, baseline: int) -> float | None:
    """(our - baseline) / max(our, baseline). Returns None when both are 0."""
    denom = max(our, baseline)
    if denom == 0:
        return None
    return (our - baseline) / denom


def _build_comparison(results: list, baseline_csv: str, output_dir: Path):
    """Join inference results with all_baselines.csv and write comparison.csv."""
    import csv as _csv

    # Load baseline keyed by graph_id
    baselines = {}
    with open(baseline_csv, newline="") as f:
        for row in _csv.DictReader(f):
            baselines[row["graph_id"]] = row

    rows = []
    for r in results:
        # graph_name like "grafo10000.38" → graph_id "10000"
        graph_id = r["graph_name"].replace("grafo", "").split(".")[0]
        if graph_id not in baselines:
            continue
        b = baselines[graph_id]
        our = r["best_xing"]
        neato = int(b["neato_xing"])
        sfdp = int(b["sfdp_xing"])
        smartgd = int(b["smartgd_xing"])

        rows.append({
            "graph_id": graph_id,
            "graph_name": r["graph_name"],
            "num_nodes": r["num_nodes"],
            "num_edges": r["num_edges"],
            "neato_xing": neato,
            "sfdp_xing": sfdp,
            "smartgd_xing": smartgd,
            "our_xing": our,
            "ratio_vs_neato": _compute_ratio(our, neato),
            "ratio_vs_sfdp": _compute_ratio(sfdp, neato),
            "ratio_vs_smartgd": _compute_ratio(smartgd, neato),
        })

    if not rows:
        print("Warning: no rows matched between results and baseline CSV.")
        return [], float("nan"), float("nan"), float("nan")

    csv_path = output_dir / "comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Comparison saved to {csv_path}")

    # Mean ratios (skip None i.e. both-zero cases)
    def _mean_ratio(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return np.mean(vals) if vals else float("nan")

    mr_neato = _mean_ratio("ratio_vs_neato")
    mr_sfdp = _mean_ratio("ratio_vs_sfdp")
    mr_smartgd = _mean_ratio("ratio_vs_smartgd")

    print("\n--- Comparison vs Baselines (neato as reference) ---")
    print(f"  vs neato:   mean ratio = {mr_neato:+.4f}  "
          f"({'better' if mr_neato < 0 else 'worse'})")
    print(f"  vs sfdp:    mean ratio = {mr_sfdp:+.4f}  "
          f"({'better' if mr_sfdp < 0 else 'worse'})")
    print(f"  vs smartgd: mean ratio = {mr_smartgd:+.4f}  "
          f"({'better' if mr_smartgd < 0 else 'worse'})")

    return rows, mr_neato, mr_sfdp, mr_smartgd


def evaluate(
    args: BaseArgs,
    policy: BasePolicy,
    output_dir: str,
    device: torch.cuda.device,
    collect_frames: bool,
    data_root: str = None,
    split: str = "test",
    baseline_csv: str = None,
    gif_fps: int = 24,
):
    from src.plot import build_nx_graph
    print(f"Device: {device}")

    # Override dataset settings for evaluation
    if data_root:
        args.graph.data_root = data_root
    args.graph.data_split = split

    print(f"Task: {args.name}  |  Model: {args.name}")
    print(f"Data: {args.graph.data_root} / {split}")

    from src.data.rome import RomeDataset
    dataset = RomeDataset(root=args.graph.data_root, split=split)
    print(f"Dataset: {len(dataset)} graphs")

    # policy = _load_policy(ckpt, args, device)

    match args:
        case DiscretePPOArgs():
            runner = _run_discrete
        case SequentialPPOArgs() | SequentialRefinementArgs():
            runner = _run_sequential
        case _:
            raise NotImplementedError(
                f"Evaluation for task '{args.name}' not implemented")

    results: List[Dict[str, float]] = []

    print(f"initial_layout: {args.env.initial_layout}")
    for i in tqdm(range(len(dataset)), desc="Evaluating"):
        graph_data = dataset[i]

        graph = build_nx_graph(graph_data)

        outputs = runner(
            policy=policy,
            graph_data=graph_data,
            device=device,
            args=args,
            collect_frames=collect_frames,
        )

        results.append({
            "graph_name": graph_data.graph_name,
            "graph": graph,
            "num_nodes": graph_data.num_nodes,
            "num_edges": graph_data.edge_index.shape[1] // 2,
            **outputs,
        })

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    keys = [x for x in df.columns if "coords" not in x and x != "frames"]
    keys.remove("graph")  # Remove non-serializable graph object
    df[keys].to_csv(out / "results.csv", index=False)

    # ── GIF rendering ──────────────────────────────────────────────────────────
    if collect_frames:
        gif_dir = out / "gifs"
        gif_dir.mkdir(exist_ok=True)
        for r in tqdm(results, desc="Rendering GIFs"):
            frames_data: List[np.ndarray] = r.get("frames")  # type: ignore[assignment]
            if not frames_data:
                continue
            graph_name: str = r.get("graph_name")  # type: ignore[assignment]
            gif_path = gif_dir / f"{Path(graph_name).stem}.gif"
            render_gif(frames_data, r["graph"], str(gif_path), fps=gif_fps)  # type: ignore[arg-type]
        print(f"GIFs saved to {gif_dir}/")
    # ── save CSV ───────────────────────────────────────────────────────────────
    # out.mkdir(parents=True, exist_ok=True)
    # csv_path = out / "results.csv"
    # with open(csv_path, "w", newline="") as f:
    #     writer = csv.DictWriter(
    #         f,
    #         fieldnames=["graph_name", "num_nodes", "num_edges", "crossings"])
    #     writer.writeheader()
    #     writer.writerows(results)
    # print(f"\nResults saved to {csv_path}")

    # ── summary ────────────────────────────────────────────────────────────────
    crossings = [r["best_xing"] for r in results]
    zero = sum(1 for x in crossings if x == 0)
    summary_lines = [
        f"Split: {split}  |  Graphs evaluated: {len(results)}",
        f"",
        f"Crossings:",
        f"  Mean:    {np.mean(crossings):.2f}",
        f"  Median:  {np.median(crossings):.1f}",
        f"  Min:     {np.min(crossings)}",
        f"  Max:     {np.max(crossings)}",
        f"  Zero:    {zero} / {len(results)} ({100*zero/len(results):.1f}%)",
    ]
    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    summary_path = out / "summary.txt"
    summary_path.write_text(summary_text)
    print(f"Summary saved to {summary_path}")

    # ── baseline comparison ────────────────────────────────────────────────────
    comparison_rows = None
    if baseline_csv:
        comparison_rows = _build_comparison(results, baseline_csv, out)

    return results, comparison_rows
