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
from src.envs.refinement import RefinementGraphEnv
from src.models.base import BasePolicy
from src.tasks.base import BaseArgs
from typing import List, Dict
from src.tasks.sequential_ppo import SequentialPPOArgs, SequentialRefinementArgs, NodeSelectRefinementArgs
from src.tasks.discrete_ppo import DiscretePPOArgs
from src.tasks.continuous_ppo import ContinuousPPOArgs
from src.data.data import GraphData
import pandas as pd
import networkx as nx
# ── helpers ────────────────────────────────────────────────────────────────────

# ── per-graph runners ──────────────────────────────────────────────────────────


def _make_env(graph_data: GraphData, device, args: BaseArgs) -> BaseGraphEnv:
    from src.envs.continuous import ContinuousGraphEnv
    import copy
    match args:
        case DiscretePPOArgs():
            eval_config = copy.copy(args.env)
            eval_config.min_effective_action = 0.0  # no static truncation during evaluation
            eval_config.patience = 0  # run full max_steps during evaluation
            return DiscreteGraphEnv(graph_data=graph_data,
                                    device=device,
                                    config=eval_config)
        case ContinuousPPOArgs():
            return ContinuousGraphEnv(graph_data=graph_data,
                                      device=device,
                                      config=args.env)
        case SequentialPPOArgs():
            return SequentialGraphEnv(graph_data=graph_data,
                                      device=device,
                                      config=args.env)
        case SequentialRefinementArgs():
            return RefinementGraphEnv(graph_data=graph_data,
                                      device=device,
                                      config=args.env)
        case NodeSelectRefinementArgs():
            return RefinementGraphEnv(graph_data=graph_data,
                                      device=device,
                                      config=args.env)
        case _:
            raise NotImplementedError(f"No env for args type {type(args)}")


def _get_action(policy: BasePolicy, obs: np.ndarray, env: BaseGraphEnv, device,
                args: BaseArgs):
    """Call policy with the correct signature depending on env type."""
    if isinstance(args, SequentialPPOArgs):
        # Transformer policy: obs only
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
        action, _, _ = policy.get_action(obs_t, deterministic=True)
    else:
        # GAT policy: obs + edge_index + edge_attr
        node_features = torch.tensor(obs, dtype=torch.float32, device=device)
        gd = env.get_graph_data()
        action, _, _ = policy.get_action(
            node_features,
            gd["edge_index"].to(device),
            gd["edge_attr"].to(device),
            deterministic=True,
        )
    return action


def run_episode(
    policy: BasePolicy,
    graph_data: GraphData,
    device,
    args: BaseArgs,
    collect_frames: bool,
) -> Dict:
    env = _make_env(graph_data, device, args)
    before_coords = graph_data.neato_coords.numpy()
    initial_xing = graph_data.neato_xing

    obs, _ = env.reset()
    best_xing = initial_xing
    best_coords = before_coords.copy()
    frames: List[np.ndarray] = [env.get_coords().copy()
                                ] if collect_frames else []
    crossings: List[float] = [initial_xing]
    rewards = [0]
    done = False
    info: Dict = {}

    while not done:
        with torch.no_grad():
            action = _get_action(policy, obs, env, device, args)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if collect_frames:
            frames.append(env.get_coords().copy())
        current_xing = info["crossings"]
        crossings.append(current_xing)
        rewards.append(reward)
        if current_xing < best_xing:
            best_xing = current_xing
            best_coords = env.get_coords().copy()

    # GATGraphEnv subclasses expose .graph; SequentialGraphEnv exposes ._nx_graph
    graph = getattr(env, "graph", None) or getattr(env, "_nx_graph")

    # For sequential placement (one-shot), intermediate best is meaningless —
    # the final complete layout is the only valid result.
    is_sequential = isinstance(args, SequentialPPOArgs)
    if is_sequential:
        best_coords = env.get_coords().copy()
        best_xing = int(crossings[-1])

    final_xing = int(crossings[-1])

    return {
        "graph": graph,
        "coords": best_coords,
        "before_coords": before_coords,
        "initial_xing": initial_xing,
        "best_xing": best_xing,
        "final_xing": final_xing,
        "improvement": initial_xing - best_xing,
        "final_improvement": initial_xing - final_xing,
        "crossings": crossings,
        "frames": frames,
        "rewards": rewards,
        "has_best": not is_sequential,
    }


# ── GIF rendering ─────────────────────────────────────────────────────────────

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
            "ratio_vs_our": _compute_ratio(our, neato),
            "ratio_vs_sfdp": _compute_ratio(sfdp, neato),
            "ratio_vs_smartgd": _compute_ratio(smartgd, neato),
        })

    if not rows:
        print(
            "Warning: no rows matched between results and baselicase ContinuousPPOArgs():ne CSV."
        )
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

    mr_our = _mean_ratio("ratio_vs_our")
    mr_sfdp = _mean_ratio("ratio_vs_sfdp")
    mr_smartgd = _mean_ratio("ratio_vs_smartgd")

    comparison_lines = [
        "\n--- Comparison vs Baselines (neato as reference) ---",
        f"ours:   mean ratio = {mr_our:+.4f}  "
        f"({'better' if mr_our < 0 else 'worse'})",
        f"sfdp:    mean ratio = {mr_sfdp:+.4f}  "
        f"({'better' if mr_sfdp < 0 else 'worse'})",
        f" smartgd: mean ratio = {mr_smartgd:+.4f}  "
        f"({'better' if mr_smartgd < 0 else 'worse'})",
    ]
    print("\n".join(comparison_lines))

    return rows, mr_our, mr_sfdp, mr_smartgd, comparison_lines


def evaluate(
    args: BaseArgs,
    policy: BasePolicy,
    output_dir: str,
    device: torch.cuda.device,
    collect_frames: bool,
    data_root: str = None,
    split: str = "test",
    baseline_csv: str = None,
):
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

    results: List[Dict[str, float]] = []

    print(f"initial_layout: {args.env.initial_layout}")
    for i in tqdm(range(len(dataset)), desc="Evaluating"):
        graph_data = dataset[i]

        outputs = run_episode(
            policy=policy,
            graph_data=graph_data,
            device=device,
            args=args,
            collect_frames=collect_frames,
        )

        results.append({
            "graph_name": graph_data.graph_name,
            "num_nodes": graph_data.num_nodes,
            "num_edges": graph_data.edge_index.shape[1] // 2,
            "node_ids": graph_data.node_ids,
            **outputs,
        })

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    keys = [
        x for x in df.columns
        if "coords" not in x and x not in ("frames", "graph", "node_ids")
    ]
    df[keys].to_csv(out / "results.csv", index=False)

    # ── save best_coords per graph ─────────────────────────────────────────────
    coords_dir = out / "coords"
    coords_dir.mkdir(exist_ok=True)
    for r in results:
        fname = coords_dir / f"{r['graph_name']}.coord"
        node_ids = r.get("node_ids")
        coords = r["coords"]
        if node_ids is not None:
            # sort rows by numeric part of node ID (n0, n1, n2, ..., n10, n11, ...)
            rows = sorted(zip(node_ids, coords), key=lambda t: int(t[0][1:]))
            with open(fname, "w") as f:
                for nid, (x, y) in rows:
                    f.write(f"{nid} {x:.6f} {y:.6f}\n")
        else:
            np.savetxt(fname, coords, fmt="%.6f")

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

    # ── summary ─────────────────────────
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
    best_xings = np.array([r["best_xing"] for r in results])
    final_xings = np.array([r["final_xing"] for r in results])
    initial_xings = np.array([r["initial_xing"] for r in results])
    zero_best = int((best_xings == 0).sum())
    zero_final = int((final_xings == 0).sum())

    # Compute ratio vs neato if baseline_csv provided
    neato_ratio_lines = []
    if baseline_csv:
        import csv as _csv
        baselines = {}
        with open(baseline_csv, newline="") as f:
            for row in _csv.DictReader(f):
                baselines[row["graph_id"]] = int(row["neato_xing"])
        neato_vals, best_vals, final_vals = [], [], []
        for r in results:
            graph_id = r["graph_name"].replace("grafo", "").split(".")[0]
            if graph_id in baselines:
                n = baselines[graph_id]
                neato_vals.append(n)
                best_vals.append(r["best_xing"])
                final_vals.append(r["final_xing"])
        if neato_vals:
            neato_arr = np.array(neato_vals)
            best_arr = np.array(best_vals)
            final_arr = np.array(final_vals)
            denom = np.maximum(np.maximum(best_arr, neato_arr), 1)
            ratio_best = np.mean((best_arr - neato_arr) / denom)
            denom2 = np.maximum(np.maximum(final_arr, neato_arr), 1)
            ratio_final = np.mean((final_arr - neato_arr) / denom2)
            neato_ratio_lines = [
                f"",
                f"Ratio vs neato  (negative = better than neato):",
                f" best_xing:   {ratio_best:+.4f}  ({'better' if ratio_best < 0 else 'worse'})",
                f"  final_xing:  {ratio_final:+.4f}  ({'better' if ratio_final < 0 else 'worse'})",
            ]

    summary_lines = [
        f"Split: {split}  |  Graphs evaluated: {len(results)}",
        f"",
        f"Best crossings (lowest seen during episode):",
        f"  Mean:    {np.mean(best_xings):.2f}",
        f"  Median:  {np.median(best_xings):.1f}",
        f"  Min:     {int(np.min(best_xings))}",
        f"  Max:     {int(np.max(best_xings))}",
        f"  Zero:    {zero_best} / {len(results)} ({100*zero_best/len(results):.1f}%)",
        f"",
        f"Final crossings (end-of-episode state):",
        f"  Mean:    {np.mean(final_xings):.2f}",
        f"  Median:  {np.median(final_xings):.1f}",
        f"  Min:     {int(np.min(final_xings))}",
        f"  Max:     {int(np.max(final_xings))}",
        f"  Zero:    {zero_final} / {len(results)} ({100*zero_final/len(results):.1f}%)",
        *neato_ratio_lines,
    ]
    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    # ── config dump ────────────────────────────────────────────────────────────
    import dataclasses, json as _json
    config_dict = dataclasses.asdict(args)
    config_lines = "\n\n--- Config ---\n" + _json.dumps(
        config_dict, indent=2, default=str)

    summary_path = out / "summary.txt"
    summary_path.write_text(summary_text + config_lines)
    print(f"Summary saved to {summary_path}")

    # ── baseline comparison ────────────────────────────────────────────────────
    comparison_rows = None
    if baseline_csv:
        comparison_rows = _build_comparison(results, baseline_csv, out)
        _, _, _, _, comparison_lines = comparison_rows
        with open(summary_path, "a") as f:
            f.write("\n" + "\n".join(comparison_lines) + "\n")

    return results, comparison_rows
