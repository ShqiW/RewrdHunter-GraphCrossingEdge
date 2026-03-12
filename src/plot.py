"""
Plot before/after graph layouts for top-N improved test graphs.

Called from main.py if_plot branch, or standalone via 23_visualize_layouts.py.
"""
import torch
import numpy as np
import networkx as nx
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from tqdm import tqdm

# ── Graph helpers ─────────────────────────────────────────────────────────────


def build_nx_graph(graph_data) -> nx.Graph:
    """Build nx.Graph from GraphData."""
    graph = nx.Graph()
    graph.add_nodes_from(range(graph_data.num_nodes))
    edges = graph_data.edge_index.T[:graph_data.edge_index.shape[1] // 2]
    graph.add_edges_from(edges.tolist())
    return graph


def count_hard_crossings(graph: nx.Graph, coords: np.ndarray, device) -> int:
    """Count true (integer) crossings using hard XingLoss."""
    from src.losses.xing import XingLoss
    xing = XingLoss(graph, device, soft=False)
    return int(xing(torch.tensor(coords, dtype=torch.float32)).item())


def segments_intersect(p1, p2, p3, p4) -> bool:

    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(
        p1, p2, p4))


def get_crossing_edges(coords: np.ndarray, edges: list) -> set:
    """Return set of edge indices that participate in at least one crossing."""
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


# ── Episode runner ────────────────────────────────────────────────────────────


def run_episode(policy, env, graph: nx.Graph, device, seed: int = 42):
    """
    Run one full episode.

    Returns:
        before_coords: initial coordinates
        best_coords:   coordinates at best (minimum) soft crossing during episode
        initial_xing:  hard crossing count at start
        best_xing:     hard crossing count at best coords
    """
    np.random.seed(seed)
    obs, info = env.reset(seed=seed)
    before_coords = env.get_coords()

    best_soft = info["crossings"]
    best_coords = before_coords.copy()

    done = False
    while not done:
        node_features = torch.tensor(obs, dtype=torch.float32, device=device)
        graph_data_dict = env.get_graph_data()
        edge_index = graph_data_dict["edge_index"].to(device)
        edge_attr = graph_data_dict["edge_attr"].to(device)

        with torch.no_grad():
            action, _, _ = policy.get_action(node_features,
                                             edge_index,
                                             edge_attr,
                                             deterministic=True)

        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if info["crossings"] < best_soft:
            best_soft = info["crossings"]
            best_coords = env.get_coords()

    initial_xing = count_hard_crossings(graph, before_coords, device)
    best_xing = count_hard_crossings(graph, best_coords, device)

    return before_coords, best_coords, initial_xing, best_xing


# ── Drawing ───────────────────────────────────────────────────────────────────


def draw_graph(ax, coords: np.ndarray, edges: list, crossing_edges: set,
               title: str):
    ax.set_aspect("equal")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
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


# ── Main entry ────────────────────────────────────────────────────────────────


def plot_layouts(args):
    """
    Run inference on test set and save before/after visualization.

    Args:
        args: DiscretePPOArgs (or any BaseArgs with env/model/graph/save_path)
        checkpoint_path: override checkpoint path; if None, uses args.save_path
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Resolve checkpoint
    save_path = Path(args.save_path)
    ckpt_path = save_path / "final_model.pt"
    if not ckpt_path.exists():
        checkpoints = sorted(save_path.glob("checkpoint_*.pt"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoint found in {save_path}")
        ckpt_path = checkpoints[-1]

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    from src.models.gnn import DiscreteGNNPolicy
    policy = DiscreteGNNPolicy(config=args.model).to(device)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()

    # Load test dataset
    from src.data.rome import RomeDataset
    from src.envs.discrete import DiscreteGraphEnv

    dataset = RomeDataset(root=args.graph.data_root, split="test")
    print(f"Test dataset: {len(dataset)} graphs")

    results = []
    for i in tqdm(range(len(dataset)), desc="Running inference"):
        graph_data = dataset[i]
        graph = build_nx_graph(graph_data)
        env = DiscreteGraphEnv(graph_data=graph_data,
                               device=device,
                               config=args.env)
        try:
            before_coords, best_coords, init_xing, best_xing = run_episode(
                policy, env, graph, device)
            results.append({
                "graph_name": graph_data.graph_name,
                "graph": graph,
                "before_coords": before_coords,
                "after_coords": best_coords,
                "initial_xing": init_xing,
                "best_xing": best_xing,
                "improvement": init_xing - best_xing,
            })
        except Exception as e:
            print(f"  Skipping {graph_data.graph_name}: {e}")

    # Select top-N by improvement

    results.sort(key=lambda r: r["improvement"], reverse=True)
    top = results
    n = len(top)
    # Plot
    output = Path(args.save_path) / "layout_visualization.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(n, 2, figsize=(8, n * 3.2))
    fig.suptitle(
        "Graph Layout: Before vs. After PPO Optimization\n(red = crossing edges)",
        fontsize=12,
        y=1.01)

    for row, r in enumerate(top):
        edges = list(r["graph"].edges())
        before_xing_edges = get_crossing_edges(r["before_coords"], edges)
        after_xing_edges = get_crossing_edges(r["after_coords"], edges)

        draw_graph(
            axes[row, 0], r["before_coords"], edges, before_xing_edges,
            f"{r['graph_name']}\nBefore  crossings={r['initial_xing']}")
        draw_graph(
            axes[row, 1], r["after_coords"], edges, after_xing_edges,
            f"After PPO  crossings={r['best_xing']}  (Δ={r['improvement']})")

    legend_patches = [
        mpatches.Patch(color="#e74c3c", label="Crossing edge"),
        mpatches.Patch(color="#aaaaaa", label="Normal edge"),
    ]
    fig.legend(handles=legend_patches,
               loc="lower center",
               ncol=2,
               bbox_to_anchor=(0.5, -0.01),
               fontsize=9)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"\nSaved to {output}")
