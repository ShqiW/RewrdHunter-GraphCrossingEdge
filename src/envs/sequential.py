"""
Sequential node placement environment for graph layout optimization.

Nodes are placed one by one in BFS order onto an empty canvas.
Action: continuous 2D coordinate (x, y) ∈ [0, 1]²
Reward: -new_crossings(v_t) + λ·ΔStructure

State (for deciding where to place v_t):
    node_features: [num_placed, 3] — (x, y, is_neighbor_of_vt)
    edge_index:    [2, num_placed_edges] — topology of placed subgraph
"""
from dataclasses import dataclass
from collections import deque
from typing import Optional

import numpy as np
import networkx as nx
import torch
from gymnasium import spaces

from src.tasks.base import BaseEnvConfig
from src.envs.base import BaseGraphEnv
from src.data.rome import GraphData
from src.envs.utils import get_initial_layout
# Acceleration priority: Cython compiled > Numba JIT > pure numpy (defined below)
try:
    from src.envs._crossing import segments_intersect_batch as _fast_intersect
    _BACKEND = "cython"
    print(f"Using Cython-accelerated crossing checker (_crossing.pyx)")
except ImportError:
    try:
        from src.envs._crossing_numba import segments_intersect_batch as _fast_intersect
        _BACKEND = "numba"
        print(f"Using Numba-accelerated crossing checker (_crossing_numba.py)")
    except ImportError:
        _fast_intersect = None
        print(
            "Using pure NumPy crossing checker (slow; install Cython or Numba for speed)"
        )
        _BACKEND = "numpy"


@dataclass
class SequentialEnvConfig(BaseEnvConfig):
    """Sequential placement environment configuration"""
    structure_weight: float = 0.3
    normalize_coords: bool = False  # re-normalize placed coords to [-1,1]² after each step
    # Node ordering strategy:
    #   "bfs"          — BFS from random start (default)
    #   "dfs"          — DFS from random start
    #   "random"       — uniform random permutation
    #   "degree_desc"  — highest-degree nodes first
    #   "degree_asc"   — lowest-degree nodes first
    #   "degree_sample"— sample without replacement; P(v) ∝ softmax(degree)
    order_method: str = "bfs"
    rotate_augment: bool = False  # randomly rotate all placed coords after each step
    # Initial layout for refinement mode:
    #   "none"   — place from scratch (default)
    #   "neato"  — start from graphviz neato layout, action = delta offset
    #   "sfdp"   — start from graphviz sfdp layout
    #   "spring" — start from networkx spring layout
    # initial_layout: str = "none"
    delta_scale: float = 0.1  # max offset per step in [-1,1]² space (only used when initial_layout != "none")


def _segments_intersect_batch(
    p1: np.ndarray,
    p2: np.ndarray,
    p3s: np.ndarray,
    p4s: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Check if segment (p1, p2) intersects each segment (p3s[i], p4s[i]).

    Two-phase approach borrowed from polygon intersection detection:
      Phase 1 — AABB pre-filter (vectorised, cheap): discard pairs whose
                 bounding boxes do not overlap — necessary condition for intersection.
      Phase 2 — Exact cross-product test (only on AABB survivors).

    Args:
        p1, p2: shape (2,) — endpoints of the new edge
        p3s, p4s: shape (M, 2) — endpoints of M placed edges

    Returns:
        bool array of shape (M,)
    """
    result = np.zeros(len(p3s), dtype=bool)

    # ── Phase 1: AABB pre-filter ────────────────────────────────────────────
    p1_min = np.minimum(p1, p2)  # (2,)
    p1_max = np.maximum(p1, p2)  # (2,)
    p3_min = np.minimum(p3s, p4s)  # (M, 2)
    p3_max = np.maximum(p3s, p4s)  # (M, 2)

    aabb = ((p1_max[0] >= p3_min[:, 0]) & (p1_min[0] <= p3_max[:, 0]) &
            (p1_max[1] >= p3_min[:, 1]) & (p1_min[1] <= p3_max[:, 1]))
    if not aabb.any():
        return result

    # ── Phase 2: exact test on AABB survivors only ──────────────────────────
    p3c = p3s[aabb]
    p4c = p4s[aabb]

    r = p2 - p1
    s = p4c - p3c
    qmp = p3c - p1

    rxs = r[0] * s[:, 1] - r[1] * s[:, 0]
    qmpxr = qmp[:, 0] * r[1] - qmp[:, 1] * r[0]
    qmpxs = qmp[:, 0] * s[:, 1] - qmp[:, 1] * s[:, 0]

    parallel = np.abs(rxs) < 1e-10
    safe_rxs = np.where(parallel, 1.0, rxs)

    t = qmpxs / safe_rxs
    u = qmpxr / safe_rxs

    proper = (~parallel) & (t > eps) & (t < 1 - eps) & (u > eps) & (u
                                                                    < 1 - eps)

    # Collinear overlap: closes the degenerate "all-nodes-on-a-line" trick
    collinear = parallel & (np.abs(qmpxr) < 1e-10)
    r_sq = float(r[0]**2 + r[1]**2)
    safe_r_sq = r_sq if r_sq > 1e-10 else 1.0
    t3 = (qmp * r).sum(axis=1) / safe_r_sq
    t4 = t3 + (s * r).sum(axis=1) / safe_r_sq
    t_lo = np.minimum(t3, t4)
    t_hi = np.maximum(t3, t4)
    collinear_overlap = collinear & (t_hi > eps) & (t_lo < 1 -
                                                    eps) & (t_hi - t_lo > eps)

    result[aabb] = proper | collinear_overlap
    return result


def _segments_intersect_batch_dispatch(p1, p2, p3s, p4s, eps=1e-6):
    """Route to the fastest available backend."""
    if _fast_intersect is not None:
        return _fast_intersect(p1, p2, p3s, p4s, eps)
    return _segments_intersect_batch(p1, p2, p3s, p4s, eps)


class SequentialGraphEnv(BaseGraphEnv):
    """
    Sequential node placement environment.

    Episode flow:
        reset()  → empty obs (no nodes placed), info with next vt
        step(xy) → place vt at xy, compute reward, return new obs
        ...
        step(xy) → place last node, terminated=True

    The observation is always the CURRENT placed state used to decide
    where to place the NEXT node (vt).
    """

    def __init__(
        self,
        graph_data: GraphData,
        device,
        config: SequentialEnvConfig,
    ):
        super().__init__()

        self.config = config
        self.device = device
        self.structure_weight = config.structure_weight
        self.normalize_coords = config.normalize_coords
        self.order_method = config.order_method
        self.rotate_augment = config.rotate_augment

        self.neato_coords = graph_data.neato_coords.numpy()
        # Build graph from GraphData
        self.num_nodes = graph_data.num_nodes
        self.graph_name = graph_data.graph_name

        edge_index_np = graph_data.edge_index.numpy()  # [2, 2E]
        self.adj = [[] for _ in range(self.num_nodes)]
        self.undirected_edges = []  # list of (u, v) with u < v
        seen = set()
        for i in range(edge_index_np.shape[1]):
            u, v = int(edge_index_np[0, i]), int(edge_index_np[1, i])
            if u < v and (u, v) not in seen:
                seen.add((u, v))
                self.undirected_edges.append((u, v))
                self.adj[u].append(v)
                self.adj[v].append(u)

        self.num_edges = len(self.undirected_edges)

        # Structure reward
        self.graph_distance = graph_data.graph_distance.numpy().astype(
            np.float32)  # [n, n]
        self.tau = graph_data.tau

        # Build nx.Graph for BFS
        self._nx_graph = nx.Graph()
        self._nx_graph.add_nodes_from(range(self.num_nodes))
        self._nx_graph.add_edges_from(self.undirected_edges)

        # Gymnasium spaces
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2, ),
            dtype=np.float32,
        )
        # Variable-size obs; gymnasium needs a fixed shape declaration.
        # Actual obs tensors are variable; trainers should use get_obs_tensors().
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_nodes, 4),
            dtype=np.float32,
        )

        self.initial_layout = config.initial_layout
        self.delta_scale = config.delta_scale

        # Episode state (initialised in reset)
        self.coords: Optional[np.ndarray] = None  # [num_nodes, 2]
        self.bfs_order: Optional[list] = None
        self.placed_set: Optional[set] = None
        self.placed_nodes_list: Optional[list] = None  # ordered list
        self.placed_edges: Optional[list] = None  # list of (u,v), u<v
        self.step_idx: int = 0
        self.total_crossings: int = 0
        self.current_stress: float = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_initial_layout(self) -> np.ndarray:
        coords = get_initial_layout(
            self.initial_layout,
            self.num_nodes,
            self.neato_coords,
            self._nx_graph,
        )

        # Normalize to [-1, 1]²
        lo, hi = coords.min(axis=0), coords.max(axis=0)
        scale = hi - lo
        scale[scale == 0] = 1.0
        return (2.0 * (coords - lo) / scale - 1.0).astype(np.float32)

    def _compute_node_order(self) -> list:
        """Compute node placement order according to self.order_method."""
        method = self.order_method
        n = self.num_nodes
        start = int(np.random.randint(n))

        if method == "bfs":
            return self._bfs_from(start)

        if method == "dfs":
            return self._dfs_from(start)

        if method == "random":
            perm = list(range(n))
            np.random.shuffle(perm)
            return perm

        if method in ("degree_desc", "degree_asc"):
            degrees = [len(self.adj[v]) for v in range(n)]
            reverse = (method == "degree_desc")
            return sorted(range(n), key=lambda v: degrees[v], reverse=reverse)

        if method == "degree_sample":
            degrees = np.array([len(self.adj[v]) for v in range(n)],
                               dtype=np.float32)
            # softmax to get probabilities
            d = degrees - degrees.max()
            probs = np.exp(d)
            probs /= probs.sum()
            return list(np.random.choice(n, size=n, replace=False, p=probs))

        raise ValueError(f"Unknown order_method: {method!r}")

    def _bfs_from(self, start: int) -> list:
        order, visited = [], set()
        queue = deque([start])
        visited.add(start)
        while queue:
            node = queue.popleft()
            order.append(node)
            for nb in self.adj[node]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        for n in range(self.num_nodes):
            if n not in visited:
                order.append(n)
        return order

    def _dfs_from(self, start: int) -> list:
        order, visited = [], set()
        stack = [start]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            order.append(node)
            for nb in reversed(self.adj[node]
                               ):  # reversed to preserve left-to-right visit
                if nb not in visited:
                    stack.append(nb)
        for n in range(self.num_nodes):
            if n not in visited:
                order.append(n)
        return order

    def _count_new_crossings(self, vt: int, pos: np.ndarray) -> int:
        """
        Count edge crossings introduced by placing vt at pos.
        Only checks new edges (vt, u) vs all already-placed edges.
        """
        placed_neighbors = [u for u in self.adj[vt] if u in self.placed_set]
        if not placed_neighbors or not self.placed_edges:
            return 0

        # Build placed-edge arrays once (O(m) numpy, not O(k*m) Python loops)
        edge_arr = np.array(self.placed_edges, dtype=np.int32)  # [m, 2]
        eu, ev = edge_arr[:, 0], edge_arr[:, 1]
        all_p3 = self.coords[eu]  # [m, 2]
        all_p4 = self.coords[ev]  # [m, 2]

        total = 0
        for nb in placed_neighbors:
            # Skip edges adjacent to (vt, nb): those sharing endpoint nb.
            # vt has no placed edges yet so no need to filter for vt.
            valid = (eu != nb) & (ev != nb)
            if not valid.any():
                continue
            # Degenerate case: zero-length edge cannot be tested geometrically.
            # Treat it as crossing every valid placed edge (worst case), so the
            # policy cannot exploit stacking to achieve zero crossing count.
            if np.linalg.norm(pos - self.coords[nb]) < 1e-6:
                total += int(valid.sum())
                continue
            hits = _segments_intersect_batch_dispatch(
                pos,
                self.coords[nb],
                all_p3[valid],
                all_p4[valid],
            )
            total += int(hits.sum())

        return total

    def _build_obs(self):
        """
        Build observation for deciding where to place bfs_order[step_idx].

        Returns:
            node_features: np.ndarray [num_placed, 3] — (x, y, is_neighbor_of_vt)
            placed_nodes:  list of node indices in the same row order
        """
        num_placed = len(self.placed_nodes_list)
        if num_placed == 0:
            return np.zeros((0, 4), dtype=np.float32), []

        vt = self.bfs_order[self.step_idx]
        placed_neighbors = set(u for u in self.adj[vt] if u in self.placed_set)

        features = np.zeros((num_placed, 4), dtype=np.float32)
        for i, node in enumerate(self.placed_nodes_list):
            features[i, 0] = self.coords[node, 0]
            features[i, 1] = self.coords[node, 1]
            features[i, 2] = 1.0 if node == vt else 0.0
            features[i, 3] = 1.0 if node in placed_neighbors else 0.0

        return features, list(self.placed_nodes_list)

    def _compute_structure_stress(self, placed_indices: list) -> float:
        """Normalized stress over the placed subgraph.

        Stress = Σ_{i<j} w_ij * (d_layout_ij - α * d_graph_ij)² / num_pairs
        w_ij = 1 / d_graph_ij², α computed adaptively.
        Returns ~0 for good layouts, ~1 for full cluster.
        """
        k = len(placed_indices)
        if k < 2:
            return 0.0

        idx = np.array(placed_indices)

        d_graph_sub = self.graph_distance[np.ix_(idx, idx)]  # [k, k]
        coords_sub = self.coords[idx]  # [k, 2]
        diff = coords_sub[:, None, :] - coords_sub[None, :, :]  # [k, k, 2]
        d_layout_sub = np.sqrt((diff**2).sum(axis=-1) + 1e-6)  # [k, k]

        # Upper triangle only (i < j), exclude zero graph distances
        triu = np.triu(np.ones((k, k), dtype=bool), k=1)
        valid = triu & (d_graph_sub > 0)
        if not valid.any():
            return 0.0

        d_g = d_graph_sub[valid]
        d_l = d_layout_sub[valid]
        W = 1.0 / np.maximum(d_g**2, 1e-3)

        # Adaptive scale factor α (disabled: allows model to rationalize degenerate solutions)
        # denom = (W * d_l**2).sum()
        # alpha = (W * d_l * d_g).sum() / denom if denom > 1e-9 else 1.0
        # stress = (W * (d_l - alpha * d_g)**2).sum()

        stress = (W * (d_l - d_g)**2).sum()
        num_pairs = k * (k - 1) / 2
        return float(stress / num_pairs)

    def _build_edge_index(self, placed_nodes: list) -> np.ndarray:
        """Build edge_index for the placed subgraph (local indices)."""
        node_to_idx = {node: i for i, node in enumerate(placed_nodes)}
        src, dst = [], []
        for (u, v) in self.placed_edges:
            if u in node_to_idx and v in node_to_idx:
                ui, vi = node_to_idx[u], node_to_idx[v]
                src += [ui, vi]
                dst += [vi, ui]
        if src:
            return np.array([src, dst], dtype=np.int64)
        return np.zeros((2, 0), dtype=np.int64)

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.bfs_order = self._compute_node_order()
        if self.initial_layout != "none":
            self.coords = self._get_initial_layout()
        else:
            self.coords = np.zeros((self.num_nodes, 2), dtype=np.float32)
        self.placed_set = set()
        self.placed_nodes_list = []
        self.placed_edges = []
        self.step_idx = 0
        self.total_crossings = 0
        self.current_stress = 0.0

        # When normalization is enabled, auto-place the first node at a random
        # position in [-1, 1]² to serve as a fixed anchor.  The agent's action
        # for step 0 would be meaningless (normalised away), so we skip it and
        # advance step_idx here.
        if self.normalize_coords:
            v0 = self.bfs_order[0]
            anchor = self.np_random.uniform(-1.0, 1.0,
                                            size=(2, )).astype(np.float32)
            self.coords[v0] = anchor
            self.placed_set.add(v0)
            self.placed_nodes_list.append(v0)
            self.step_idx = 1

        # Obs is empty: no nodes placed yet (or only anchor placed)
        node_features = np.zeros((0, 4), dtype=np.float32)
        edge_index = np.zeros((2, 0), dtype=np.int64)

        next_step = self.step_idx
        info = {
            "vt":
            self.bfs_order[next_step] if next_step < self.num_nodes else None,
            "step": next_step,
            "total_crossings": 0,
            "node_features": node_features,
            "edge_index": edge_index,
            "placed_nodes": [],
        }
        return node_features, info

    def step(self, action: np.ndarray):
        assert self.step_idx < self.num_nodes, "Episode already finished"

        vt = self.bfs_order[self.step_idx]
        if self.initial_layout != "none":
            pos = np.clip(self.coords[vt] + action * self.delta_scale, -1.0,
                          1.0).astype(np.float32)
        else:
            pos = np.clip(action, -1.0, 1.0).astype(np.float32)

        # Crossing penalty
        new_crossings = self._count_new_crossings(vt, pos)
        self.total_crossings += new_crossings
        reward = -float(new_crossings)

        # Place node
        self.coords[vt] = pos
        self.placed_set.add(vt)
        self.placed_nodes_list.append(vt)

        # Register new placed edges
        for nb in self.adj[vt]:
            if nb in self.placed_set and nb != vt:
                self.placed_edges.append((min(vt, nb), max(vt, nb)))

        # Dynamic normalization: rescale all placed coords to [-1, 1]²
        if self.normalize_coords and len(self.placed_nodes_list) >= 2:
            placed = np.array(self.placed_nodes_list)
            c = self.coords[placed]
            lo, hi = c.min(axis=0), c.max(axis=0)
            scale = hi - lo
            scale[scale == 0] = 1.0
            self.coords[placed] = 2.0 * (c - lo) / scale - 1.0

        # Random rotation augmentation: rotate all placed coords around their
        # centroid by a uniform random angle, then re-normalize to [-1, 1]².
        # Crossing count is invariant to rotation; this increases layout diversity.
        if self.rotate_augment and len(self.placed_nodes_list) >= 2:
            placed = np.array(self.placed_nodes_list)
            c = self.coords[placed]
            theta = self.np_random.uniform(0.0, 2.0 * np.pi)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            centroid = c.mean(axis=0)
            cc = c - centroid
            rotated = np.stack([
                cos_t * cc[:, 0] - sin_t * cc[:, 1],
                sin_t * cc[:, 0] + cos_t * cc[:, 1],
            ],
                               axis=1).astype(np.float32)
            lo, hi = rotated.min(axis=0), rotated.max(axis=0)
            scale = hi - lo
            scale[scale == 0] = 1.0
            self.coords[placed] = 2.0 * (rotated - lo) / scale - 1.0

        if (self.structure_weight > 0):
            # Structure reward: penalize delta stress (marginal cost of this placement)
            new_stress = self._compute_structure_stress(self.placed_nodes_list)
            delta_stress = new_stress - self.current_stress
            self.current_stress = new_stress
            reward -= self.structure_weight * delta_stress

        self.step_idx += 1
        terminated = (self.step_idx >= self.num_nodes)
        truncated = self.step_idx >= self.config.max_steps

        if terminated:
            node_features = np.zeros((0, 4), dtype=np.float32)
            edge_index = np.zeros((2, 0), dtype=np.int64)
            placed_nodes = []
        else:
            node_features, placed_nodes = self._build_obs()
            edge_index = self._build_edge_index(placed_nodes)

        info = {
            "vt": vt,
            "step": self.step_idx,
            "new_crossings": new_crossings,
            "total_crossings": self.total_crossings,
            "node_features": node_features,
            "edge_index": edge_index,
            "placed_nodes": placed_nodes,
        }
        return node_features, reward, terminated, truncated, info

    def get_graph_data(self):
        """Return current state tensors (used by trainer to feed policy)."""
        if not self.placed_nodes_list or self.step_idx >= self.num_nodes:
            return {
                "node_features": np.zeros((0, 4), dtype=np.float32),
                "edge_index": np.zeros((2, 0), dtype=np.int64),
                "placed_nodes": [],
            }
        node_features, placed_nodes = self._build_obs()
        edge_index = self._build_edge_index(placed_nodes)
        return {
            "node_features": node_features,
            "edge_index": edge_index,
            "placed_nodes": placed_nodes,
        }

    def get_coords(self) -> np.ndarray:
        return self.coords.copy()

    # ── Policy-input interface ─────────────────────────────────────────────────

    def get_policy_input(self, obs: np.ndarray, device) -> tuple:
        """Return (node_features_tensor,) for policy.get_action."""
        return (torch.tensor(obs, dtype=torch.float32, device=device), )

    @staticmethod
    def make_batch_input(envs, obs_list, device):
        """
        Build a list of node-feature tensors from current observations.
        Used by policy.get_action_batched(batch_input) during rollout.
        """
        return [
            torch.tensor(obs, dtype=torch.float32, device=device)
            for obs in obs_list
        ]
