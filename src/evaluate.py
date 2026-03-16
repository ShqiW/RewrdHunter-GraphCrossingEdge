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
import csv
from pathlib import Path
from tqdm import tqdm
from dataclasses import fields
import argparse

# ── helpers ────────────────────────────────────────────────────────────────────


def _load_checkpoint(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    return ckpt


def _rebuild_args(args_dict: dict):
    """Reconstruct a SequentialPPOArgs or DiscretePPOArgs from a saved dict."""
    name = args_dict.get("name", "sequential_ppo")
    if name == "sequential_ppo":
        from src.tasks.sequential_ppo import SequentialPPOArgs
        from src.envs.sequential import SequentialEnvConfig
        from src.models.transformer_policy import TransformerConfig
        from src.tasks.base import BasePPOConfig, BaseGraphConfig, BaseRewardConfig

        def _fill(cls, d):
            valid = {f.name for f in fields(cls)}
            return cls(**{k: v for k, v in d.items() if k in valid})

        obj = SequentialPPOArgs.__new__(SequentialPPOArgs)
        for f in fields(SequentialPPOArgs):
            if f.name == "env":
                setattr(obj, "env",
                        _fill(SequentialEnvConfig, args_dict.get("env", {})))
            elif f.name == "model":
                setattr(obj, "model",
                        _fill(TransformerConfig, args_dict.get("model", {})))
            elif f.name == "ppo":
                setattr(obj, "ppo",
                        _fill(BasePPOConfig, args_dict.get("ppo", {})))
            elif f.name == "graph":
                setattr(obj, "graph",
                        _fill(BaseGraphConfig, args_dict.get("graph", {})))
            elif f.name == "reward":
                setattr(obj, "reward",
                        _fill(BaseRewardConfig, args_dict.get("reward", {})))
            else:
                setattr(obj, f.name, args_dict.get(f.name, f.default))
        return obj
    else:
        from src.tasks.discrete_ppo import DiscretePPOArgs
        from src.envs.discrete import DiscreteEnvConfig
        from src.models.gnn import GNNConfig
        from src.tasks.base import BasePPOConfig, BaseGraphConfig, BaseRewardConfig

        def _fill(cls, d):
            valid = {f.name for f in fields(cls)}
            return cls(**{k: v for k, v in d.items() if k in valid})

        obj = DiscretePPOArgs.__new__(DiscretePPOArgs)
        for f in fields(DiscretePPOArgs):
            if f.name == "env":
                setattr(obj, "env",
                        _fill(DiscreteEnvConfig, args_dict.get("env", {})))
            elif f.name == "model":
                setattr(obj, "model",
                        _fill(GNNConfig, args_dict.get("model", {})))
            elif f.name == "ppo":
                setattr(obj, "ppo",
                        _fill(BasePPOConfig, args_dict.get("ppo", {})))
            elif f.name == "graph":
                setattr(obj, "graph",
                        _fill(BaseGraphConfig, args_dict.get("graph", {})))
            elif f.name == "reward":
                setattr(obj, "reward",
                        _fill(BaseRewardConfig, args_dict.get("reward", {})))
            else:
                setattr(obj, f.name, args_dict.get(f.name, f.default))
        return obj


def _load_policy(ckpt, args, device):
    from src.train import create_model
    policy = create_model(args, env=None)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.to(device).eval()
    return policy


# ── per-graph runners ──────────────────────────────────────────────────────────


def _run_sequential(policy, graph_data, args, device):
    from src.envs.sequential import SequentialGraphEnv
    from src.losses.xing import XingLoss
    import networkx as nx
    env = SequentialGraphEnv(graph_data=graph_data,
                             device=device,
                             config=args.env)
    obs, _ = env.reset()
    done = False
    while not done:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            action, _, _ = policy.get_action(obs_t, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    coords = env.get_coords()
    # Recompute final crossings with XingLoss for consistency with discrete/plot
    G = nx.Graph()
    G.add_nodes_from(range(graph_data.num_nodes))
    edges = graph_data.edge_index.T[:graph_data.edge_index.shape[1] //
                                    2].tolist()
    G.add_edges_from(edges)
    xing = XingLoss(G, device, soft=False)
    total_xing = int(
        xing(torch.tensor(coords, dtype=torch.float32, device=device)).item())
    return coords, total_xing


def _run_discrete(policy, graph_data, args, device):
    from src.envs.discrete import DiscreteGraphEnv
    env = DiscreteGraphEnv(graph_data=graph_data,
                           device=device,
                           config=args.env)
    obs, info = env.reset()
    best_xing = info["crossings"]
    best_coords = env.get_coords()
    done = False
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
        if info["crossings"] < best_xing:
            best_xing = info["crossings"]
            best_coords = env.get_coords()
    return best_coords, best_xing


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
        our = r["crossings"]
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
            "ratio_vs_sfdp": _compute_ratio(our, sfdp),
            "ratio_vs_smartgd": _compute_ratio(our, smartgd),
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
    checkpoint_path: str,
    output_dir: str,
    data_root: str = None,
    split: str = "test",
    device: str = None,
    baseline_csv: str = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    ckpt = _load_checkpoint(checkpoint_path, device)
    args = _rebuild_args(ckpt["args"])

    # Override dataset settings for evaluation
    if data_root:
        args.graph.data_root = data_root
    args.graph.data_split = split
    args.graph.use_dataset = True

    print(f"Task: {args.name}  |  Model: {args.model.type}")
    print(f"Data: {args.graph.data_root} / {split}")

    from src.data.rome import RomeDataset
    dataset = RomeDataset(root=args.graph.data_root, split=split)
    print(f"Dataset: {len(dataset)} graphs")

    policy = _load_policy(ckpt, args, device)

    is_sequential = (args.env.type == "sequential")
    runner = _run_sequential if is_sequential else _run_discrete

    results = []
    for i in tqdm(range(len(dataset)), desc="Evaluating"):
        graph_data = dataset[i]
        try:
            coords, total_xing = runner(policy, graph_data, args, device)
            results.append({
                "graph_name": graph_data.graph_name,
                "num_nodes": graph_data.num_nodes,
                "num_edges": graph_data.edge_index.shape[1] // 2,
                "crossings": int(total_xing),
            })
        except Exception as e:
            print(f"  Skipping {graph_data.graph_name}: {e}")

    # ── save CSV ───────────────────────────────────────────────────────────────
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["graph_name", "num_nodes", "num_edges", "crossings"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {csv_path}")

    # ── summary ────────────────────────────────────────────────────────────────
    crossings = [r["crossings"] for r in results]
    zero = sum(1 for x in crossings if x == 0)
    summary_lines = [
        f"Checkpoint: {checkpoint_path}",
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
