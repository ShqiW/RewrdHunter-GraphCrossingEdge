"""
Plot graph layouts for visualization.

Can be called two ways:
  1. Via main.py (if_plot=True) — uses args directly
  2. Standalone from checkpoint — reconstructs args automatically

- discrete_ppo: before (random init) vs after (PPO refinement)
- sequential_ppo: final placement produced by the policy
"""
import argparse
import torch
import numpy as np
import networkx as nx
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from tqdm import tqdm

# ── Graph helpers ──────────────────────────────────────────────────────────────


def build_nx_graph(graph_data) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(graph_data.num_nodes))
    edges = graph_data.edge_index.T[:graph_data.edge_index.shape[1] // 2]
    graph.add_edges_from(edges.tolist())
    return graph


def count_hard_crossings(graph: nx.Graph, coords: np.ndarray, device) -> int:
    from src.losses.xing import XingLoss
    xing = XingLoss(graph, device, soft=False)
    return int(xing(torch.tensor(coords, dtype=torch.float32)).item())


def segments_intersect(p1, p2, p3, p4) -> bool:

    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(
        p1, p2, p4))


def get_crossing_edges(coords: np.ndarray, edges: list) -> set:
    crossing = set()
    n = len(edges)
    for i in range(n):
        u1, v1 = edges[i]
        for j in range(i + 1, n):
            u2, v2 = edges[j]
            if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                continue
            if segments_intersect(coords[u1], coords[v1], coords[u2],
                                  coords[v2]):
                crossing.add(i)
                crossing.add(j)
    return crossing


# ── Policy / args loading ──────────────────────────────────────────────────────


def load_policy_from_path(ckpt_path: str, args, device: str):
    """Load policy weights given an already-built args object."""
    ckpt = torch.load(ckpt_path, map_location=device)
    model_type = args.model.type
    if model_type == "gnn":
        from src.models.gnn import DiscreteGNNPolicy
        policy = DiscreteGNNPolicy(config=args.model).to(device)
    elif model_type == "transformer":
        from src.models.transformer_policy import TransformerPlacementPolicy
        policy = TransformerPlacementPolicy(config=args.model).to(device)
    else:
        raise NotImplementedError(f"Unknown model type: {model_type}")
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()
    return policy


def load_policy(args, device: str):
    """Load policy from args.save_path (used when called via main.py)."""
    save_path = Path(args.save_path)
    ckpt_path = save_path / "final_model.pt"
    if not ckpt_path.exists():
        checkpoints = sorted(save_path.glob("checkpoint_*.pt"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoint found in {save_path}")
        ckpt_path = checkpoints[-1]
    print(f"Loading checkpoint: {ckpt_path}")
    return load_policy_from_path(str(ckpt_path), args, device)


# ── Episode runners ────────────────────────────────────────────────────────────


def run_episode_discrete(policy, env, graph: nx.Graph, device, seed: int = 42):
    np.random.seed(seed)
    obs, info = env.reset(seed=seed)
    before_coords = env.get_coords()
    best_soft = info["crossings"]
    best_coords = before_coords.copy()
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
        if info["crossings"] < best_soft:
            best_soft = info["crossings"]
            best_coords = env.get_coords()
    initial_xing = count_hard_crossings(graph, before_coords, device)
    best_xing = count_hard_crossings(graph, best_coords, device)
    return before_coords, best_coords, initial_xing, best_xing


def run_episode_sequential(policy, env, device, seed: int = 42):
    np.random.seed(seed)
    obs, _ = env.reset(seed=seed)
    done = False
    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            action, _, _ = policy.get_action(obs_tensor, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    return env.get_coords()


# ── Drawing ────────────────────────────────────────────────────────────────────


def draw_graph(ax, coords: np.ndarray, edges: list, crossing_edges: set,
               title: str):
    ax.set_aspect("equal")
    margin = 0.05 * (coords.max() - coords.min() + 1e-6)
    ax.set_xlim(coords[:, 0].min() - margin, coords[:, 0].max() + margin)
    ax.set_ylim(coords[:, 1].min() - margin, coords[:, 1].max() + margin)
    ax.axis("off")
    ax.set_title(title, fontsize=9, pad=4)
    for idx, (u, v) in enumerate(edges):
        color = "#e74c3c" if idx in crossing_edges else "#aaaaaa"
        lw = 1.5 if idx in crossing_edges else 0.8
        ax.plot([coords[u, 0], coords[v, 0]], [coords[u, 1], coords[v, 1]],
                color=color,
                linewidth=lw,
                zorder=1)
    ax.scatter(coords[:, 0],
               coords[:, 1],
               s=30,
               c="#2c3e50",
               zorder=2,
               linewidths=0.5,
               edgecolors="white")


def _add_legend(fig):
    fig.legend(
        handles=[
            mpatches.Patch(color="#e74c3c", label="Crossing edge"),
            mpatches.Patch(color="#aaaaaa", label="Normal edge"),
        ],
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.01),
        fontsize=9,
    )


# ── Layout refinement ──────────────────────────────────────────────────────────


def refine_layout(coords: np.ndarray,
                  edges: list,
                  max_iters: int = 300) -> np.ndarray:
    """
    Post-process a layout by compressing crossing-edge nodes toward the layout
    centroid until the crossing set changes.

    Algorithm (per iteration):
      1. Find all crossing edges; collect their endpoint nodes.
      2. Binary-search the step size α ∈ (0, 1]: move each crossing node by
         α × (centroid − node) until at least one crossing is eliminated.
      3. Accept the move and repeat; stop when no α improves the layout.
    """
    coords = coords.copy().astype(np.float32)
    edge_arr = np.array(edges, dtype=np.int32)  # [m, 2]

    for _ in range(max_iters):
        crossing_set = get_crossing_edges(coords, edges)
        if not crossing_set:
            break

        n_cross = len(crossing_set)
        crossing_nodes = list(
            {node
             for idx in crossing_set
             for node in edges[idx]})
        node_idx = np.array(crossing_nodes, dtype=np.int32)

        centroid = coords.mean(axis=0)
        directions = (centroid - coords[node_idx]).astype(np.float32)  # [k, 2]

        # Binary search: find minimum α that reduces crossing count
        lo, hi = 0.0, 1.0
        best = None
        for _ in range(15):  # 2^-15 ≈ 3e-5 precision
            mid = (lo + hi) / 2.0
            trial = coords.copy()
            trial[node_idx] += mid * directions
            if len(get_crossing_edges(trial, edges)) < n_cross:
                best = trial
                hi = mid
            else:
                lo = mid

        if best is None:
            break
        coords = best

    return coords


# ── Core plot functions ────────────────────────────────────────────────────────


def _plot_sequential(policy, dataset, args, device, top_n: int, output: Path):
    from src.envs.sequential import SequentialGraphEnv

    results = []
    for i in tqdm(range(len(dataset)), desc="Running inference"):
        graph_data = dataset[i]
        graph = build_nx_graph(graph_data)
        env = SequentialGraphEnv(graph_data=graph_data,
                                 device=device,
                                 config=args.env)
        try:
            coords = run_episode_sequential(policy, env, device)
            # edges = list(graph.edges())
            # coords = refine_layout(coords, edges)
            xing = count_hard_crossings(graph, coords, device)
            results.append({
                "graph_name": graph_data.graph_name,
                "graph": graph,
                "coords": coords,
                "crossings": xing,
            })
        except Exception as e:
            print(f"  Skipping {graph_data.graph_name}: {e}")

    results.sort(key=lambda r: r["crossings"])
    subset = results[:top_n]

    out_dir = output.parent / output.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    for r in subset:
        edges = list(r["graph"].edges())
        fig, ax = plt.subplots(figsize=(5, 4.5))
        draw_graph(ax, r["coords"], edges,
                   get_crossing_edges(r["coords"], edges),
                   f"{r['graph_name']}\ncrossings={r['crossings']}")
        _add_legend(fig)
        plt.tight_layout()
        plt.savefig(out_dir / f"{r['graph_name']}.png",
                    dpi=150,
                    bbox_inches="tight")
        plt.close()

    all_xings = [r["crossings"] for r in results]
    zero_pct = 100 * sum(1 for x in all_xings if x == 0) / len(all_xings)
    print(f"\nSaved {len(subset)} figures to {out_dir}/")
    print(f"Full test  — mean: {np.mean(all_xings):.2f}, "
          f"median: {np.median(all_xings):.1f}, "
          f"zero: {sum(1 for x in all_xings if x==0)}/{len(all_xings)}, "
          f"zero-crossing={zero_pct:.1f}%")


def _plot_discrete(policy, dataset, args, device, top_n: int, output: Path):
    from src.envs.discrete import DiscreteGraphEnv

    results = []
    for i in tqdm(range(len(dataset)), desc="Running inference"):
        graph_data = dataset[i]
        graph = build_nx_graph(graph_data)
        env = DiscreteGraphEnv(graph_data=graph_data,
                               device=device,
                               config=args.env)
        try:
            before, after, init_x, best_x = run_episode_discrete(
                policy, env, graph, device)
            results.append({
                "graph_name": graph_data.graph_name,
                "graph": graph,
                "before_coords": before,
                "after_coords": after,
                "initial_xing": init_x,
                "best_xing": best_x,
                "improvement": init_x - best_x,
            })
        except Exception as e:
            print(f"  Skipping {graph_data.graph_name}: {e}")

    results.sort(key=lambda r: r["improvement"], reverse=True)
    subset = results[:top_n]

    out_dir = output.parent / output.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    for r in subset:
        edges = list(r["graph"].edges())
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        fig.suptitle(
            f"{r['graph_name']} — Before vs After PPO (red = crossing edges)",
            fontsize=10)
        draw_graph(axes[0], r["before_coords"], edges,
                   get_crossing_edges(r["before_coords"], edges),
                   f"before={r['initial_xing']}")
        draw_graph(axes[1], r["after_coords"], edges,
                   get_crossing_edges(r["after_coords"], edges),
                   f"after={r['best_xing']}  Δ={r['improvement']}")
        _add_legend(fig)
        plt.tight_layout()
        plt.savefig(out_dir / f"{r['graph_name']}.png",
                    dpi=150,
                    bbox_inches="tight")
        plt.close()

    print(f"\nSaved {len(subset)} figures to {out_dir}/")


# ── Comparison scatter plot ────────────────────────────────────────────────────


def plot_comparison(comparison_csv: str, output_dir: str):
    """
    Scatter plot of ratio vs neato for sfdp, smartgd, and ours.

    X-axis: graphs sorted by num_nodes (ascending)
    Y-axis: ratio = (method_xing - neato_xing) / max(method_xing, neato_xing)
    """
    import csv

    with open(comparison_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("plot_comparison: empty comparison CSV, skipping.")
        return

    # Sort by num_nodes ascending
    rows.sort(key=lambda r: int(r["num_nodes"]))
    x = list(range(len(rows)))

    def _ratio_vs_neato(method_key):
        vals = []
        for r in rows:
            our = int(r[method_key])
            neato = int(r["neato_xing"])
            denom = max(our, neato)
            vals.append((our - neato) / denom if denom != 0 else None)
        return vals

    ratio_sfdp = _ratio_vs_neato("sfdp_xing")
    ratio_smartgd = _ratio_vs_neato("smartgd_xing")
    ratio_ours = [
        float(r["ratio_vs_neato"])
        if r["ratio_vs_neato"] not in ("", "None") else None for r in rows
    ]

    fig, ax = plt.subplots(figsize=(max(8, len(rows) // 10), 5))

    def _scatter(ratios, label, color, marker, alpha=0.6):
        xs = [x[i] for i, v in enumerate(ratios) if v is not None]
        ys = [v for i, v in enumerate(ratios) if v is not None]
        ax.scatter(xs,
                   ys,
                   label=label,
                   color=color,
                   marker=marker,
                   s=15,
                   alpha=alpha,
                   linewidths=0)

    _scatter(ratio_sfdp, "sfdp", "#2980b9", "o")
    _scatter(ratio_smartgd, "smartgd", "#27ae60", "s")
    _scatter(ratio_ours, "ours", "#e74c3c", "^")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Graph index (sorted by num_nodes, ascending)")
    ax.set_ylabel("Ratio vs neato  (negative = better than neato)")
    ax.set_title("Per-graph ratio vs neato baseline")
    ax.legend(loc="upper left")
    plt.tight_layout()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_path = out / "comparison_scatter.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison scatter saved to {save_path}")


# ── Public entry points ────────────────────────────────────────────────────────


def plot_layouts(args, top_n: int = 100):
    """Called from main.py (if_plot=True)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = load_policy(args, device)

    from src.data.rome import RomeDataset
    dataset = RomeDataset(root=args.graph.data_root, split="test")
    print(f"Test dataset: {len(dataset)} graphs")

    output = Path().cwd() / "visualization.png"
    if args.env.type == "sequential":
        _plot_sequential(policy, dataset, args, device, top_n, output)
    else:
        _plot_discrete(policy, dataset, args, device, top_n, output)


def plot_from_checkpoint(
    checkpoint_path: str,
    output_dir: str,
    data_root: str = None,
    top_n: int = 20,
    device: str = None,
):
    """
    Standalone entry — loads everything from the checkpoint.

    Args:
        checkpoint_path: path to final_model.pt
        output_dir:      where to save layout_visualization.png
        data_root:       override data root (default: use value saved in checkpoint)
        top_n:           how many graphs to show in the figure
        device:          'cpu' or 'cuda' (default: auto)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    from src.evaluate import _rebuild_args
    ckpt = torch.load(checkpoint_path, map_location=device)
    args = _rebuild_args(ckpt["args"])

    if data_root:
        args.graph.data_root = data_root
    args.graph.data_split = "test"
    args.graph.use_dataset = True

    print(
        f"Task: {args.name}  |  Model: {args.model.type}  |  Device: {device}")
    print(f"Data: {args.graph.data_root} / test")

    policy = load_policy_from_path(checkpoint_path, args, device)

    from src.data.rome import RomeDataset
    dataset = RomeDataset(root=args.graph.data_root, split="test")
    print(f"Test dataset: {len(dataset)} graphs")

    output = Path(output_dir) / "layout_visualization.png"
    if args.env.type == "sequential":
        _plot_sequential(policy, dataset, args, device, top_n, output)
    else:
        _plot_discrete(policy, dataset, args, device, top_n, output)
