"""
Iterative layout refinement environment for graph layout optimization.

Starts with a full layout and moves nodes one by one to reduce crossings
and stress.
Action: continuous 2D delta offset (dx, dy) ∈ [-1, 1]²
Reward: -ΔCrossings − λ·ΔStress
Termination: zero crossings or max_steps reached.
"""
from dataclasses import dataclass
from typing import Optional

import networkx as nx
import numpy as np
import torch
from gymnasium import spaces

from src.data.rome import GraphData
from src.envs.base import BaseGraphEnv
from src.envs.graph_layout_state import GraphLayoutState
from src.envs.utils import get_initial_layout
from src.losses.xing import XingLoss
from src.tasks.base import BaseEnvConfig


@dataclass
class RefinementEnvConfig(BaseEnvConfig):
    """Refinement environment configuration"""
    structure_weight: float = 0.3
    exclude_non_crossing: bool = False  # focus only on nodes involved in crossings
    order_method: str = "bfs"
    delta_scale: float = 0.1  # max offset per step in [-1, 1]² space
    # Patience: truncate if crossings have not improved for this many steps.
    # Set to 0 to disable.
    patience: int = 100
    # When patience triggers, reset layout to best_coords and continue instead
    # of truncating. Episode only ends via max_steps or zero crossings.
    reset_to_best: bool = False


class RefinementGraphEnv(BaseGraphEnv):
    """
    Iterative refinement environment.

    Cycles through nodes (optionally only those in crossings) and applies
    local delta moves.  Crossing detection is routed through GraphLayoutState.
    Stress uses a precomputed O(N) incremental update.
    """

    def __init__(
        self,
        graph_data: GraphData,
        device,
        config: RefinementEnvConfig,
    ) -> None:
        super().__init__()

        self.config = config
        self.device = device
        self.structure_weight = config.structure_weight
        self.delta_scale = config.delta_scale

        self.neato_coords = graph_data.neato_coords.numpy()
        self.num_nodes = graph_data.num_nodes
        self.graph_name = graph_data.graph_name
        self.graph_distance = graph_data.graph_distance.numpy().astype(
            np.float32)
        self.tau = graph_data.tau

        # ── Build undirected edge list and adjacency ───────────────────────────
        edge_index_np = graph_data.edge_index.numpy()
        self.adj: list[list[int]] = [[] for _ in range(self.num_nodes)]
        self.undirected_edges: list[tuple[int, int]] = []
        seen: set = set()
        for i in range(edge_index_np.shape[1]):
            u, v = int(edge_index_np[0, i]), int(edge_index_np[1, i])
            if u < v and (u, v) not in seen:
                seen.add((u, v))
                self.undirected_edges.append((u, v))
                self.adj[u].append(v)
                self.adj[v].append(u)
        self.num_edges = len(self.undirected_edges)

        self._nx_graph = nx.Graph()
        self._nx_graph.add_nodes_from(range(self.num_nodes))
        self._nx_graph.add_edges_from(self.undirected_edges)

        # ── Soft crossing (optional) ───────────────────────────────────────────
        self.soft_crossing: bool = getattr(config, "soft_crossing", False)
        if self.soft_crossing:
            sharpness = getattr(config, "soft_crossing_sharpness", 10.0)
            self.xing_loss = XingLoss(self._nx_graph, device=device, soft=True,
                                      sharpness=sharpness)
        else:
            self.xing_loss = None

        # ── Stress precomputation (topology-fixed, reused every episode) ───────
        k = self.num_nodes
        triu = np.triu(np.ones((k, k), dtype=bool), k=1)
        stress_mask = triu & (self.graph_distance > 0)
        self._stress_rows, self._stress_cols = np.where(stress_mask)
        d_g_masked = self.graph_distance[self._stress_rows, self._stress_cols]
        self._stress_W = 1.0 / np.maximum(d_g_masked**2, 1e-3)
        self._stress_d_g = d_g_masked
        self._stress_norm = float(k * (k - 1) / 2) if k > 1 else 1.0

        # Per-node incremental stress: O(N) per step
        node_mask = ~np.eye(k, dtype=bool) & (self.graph_distance > 0)
        self._stress_others = [np.where(node_mask[vt])[0] for vt in range(k)]
        self._stress_d_g_from = [
            self.graph_distance[vt, self._stress_others[vt]] for vt in range(k)
        ]
        self._stress_W_from = [
            1.0 / np.maximum(d**2, 1e-3) for d in self._stress_d_g_from
        ]

        # ── Spaces ─────────────────────────────────────────────────────────────
        self.action_space = spaces.Box(low=-1.0,
                                       high=1.0,
                                       shape=(2, ),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(self.num_nodes, 5),
                                            dtype=np.float32)

        # ── Episode state (initialised in reset) ───────────────────────────────
        self.gls: GraphLayoutState
        self.coords: np.ndarray  # float64 alias to self.gls.positions
        self.node_order: list = []
        self.step_idx: int = 0
        self.current_stress: float = 0.0
        self.total_crossings: int = 0

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _get_initial_layout(self) -> np.ndarray:
        coords = get_initial_layout(
            self.config.initial_layout,
            self.num_nodes,
            self.neato_coords,
            self._nx_graph,
        )
        lo, hi = coords.min(axis=0), coords.max(axis=0)
        scale = hi - lo
        scale[scale == 0] = 1.0
        return (2.0 * (coords - lo) / scale - 1.0).astype(np.float32)

    def _get_nodes_in_crossings(self) -> list:
        """Return sorted list of node ids involved in at least one crossing."""
        _, mask = self.gls.compute_crossings(
        )  # mask: (E, E) bool, upper-triangular
        involved = np.where(mask.any(axis=1) | mask.any(axis=0))[0]
        nodes: set = set()
        for e_idx in involved:
            nodes.update(self.gls.edges[e_idx].tolist())
        return sorted(nodes)

    def _compute_structure_stress(self) -> float:
        """Full O(N²) stress — used only at reset()."""
        if self.num_nodes < 2 or len(self._stress_rows) == 0:
            return 0.0
        d_l = np.sqrt(((self.coords[self._stress_rows] -
                        self.coords[self._stress_cols])**2).sum(axis=-1) +
                      1e-6)
        return float((self._stress_W * (d_l - self._stress_d_g)**2).sum() /
                     self._stress_norm)

    def _stress_delta(self, vt: int, old_pos: np.ndarray,
                      new_pos: np.ndarray) -> float:
        """Incremental O(N) stress change when moving vt from old_pos to new_pos."""
        others = self._stress_others[vt]
        if len(others) == 0:
            return 0.0
        coords_others = self.coords[others]  # vt not updated yet
        d_g = self._stress_d_g_from[vt]
        W = self._stress_W_from[vt]
        d_old = np.sqrt(((old_pos - coords_others)**2).sum(axis=-1) + 1e-6)
        d_new = np.sqrt(((new_pos - coords_others)**2).sum(axis=-1) + 1e-6)
        return float((W * ((d_new - d_g)**2 - (d_old - d_g)**2)).sum() /
                     self._stress_norm)

    def _compute_soft_crossings(self, coords: np.ndarray) -> float:
        return self.xing_loss(
            torch.tensor(coords, dtype=torch.float32, device=self.device)
        ).item()

    def _build_obs(self) -> np.ndarray:
        vt = self.node_order[self.step_idx % len(self.node_order)]
        neighbors = set(self.adj[vt])
        crossing_nodes = set(self._get_nodes_in_crossings())
        features = np.zeros((self.num_nodes, 5), dtype=np.float32)
        features[:, 0:2] = self.coords
        features[vt, 2] = 1.0
        for nb in neighbors:
            features[nb, 3] = 1.0
        for cn in crossing_nodes:
            features[cn, 4] = 1.0
        return features

    def _build_edge_index(self) -> np.ndarray:
        src, dst = [], []
        for u, v in self.undirected_edges:
            src += [u, v]
            dst += [v, u]
        return (np.array([src, dst], dtype=np.int64) if src else np.zeros(
            (2, 0), dtype=np.int64))

    # ── Gymnasium interface ────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        initial_coords = self._get_initial_layout()
        self.gls = GraphLayoutState(self._nx_graph, range(self.num_nodes),
                                    initial_coords)
        self.coords = self.gls.positions  # float64 alias

        self.node_order = list(range(self.num_nodes))
        np.random.shuffle(self.node_order)

        if self.config.exclude_non_crossing:
            crossing_nodes = self._get_nodes_in_crossings()
            if crossing_nodes:
                self.node_order = [
                    v for v in self.node_order if v in crossing_nodes
                ]

        self.step_idx = 0
        self.current_stress = self._compute_structure_stress()
        self.total_crossings = int(self.gls.compute_crossings()[0])
        self.best_crossings = self.total_crossings
        self.best_coords = self.coords.copy()
        self.no_improve_steps = 0
        self.current_soft_crossings = (
            self._compute_soft_crossings(self.coords)
            if self.soft_crossing else float(self.total_crossings)
        )

        return self._build_obs(), {
            "vt": self.node_order[0] if self.node_order else None,
            "step": 0,
            "total_crossings": self.total_crossings,
        }

    def step(self, action: np.ndarray):
        if not self.node_order or self.step_idx >= self.config.max_steps:
            return self._build_obs(), 0.0, False, True, {
                "step": self.step_idx,
                "total_crossings": self.total_crossings,
            }

        vt = self.node_order[self.step_idx % len(self.node_order)]
        old_pos = self.gls.positions[vt].copy()
        new_pos = (old_pos + action * self.delta_scale).astype(np.float64)

        # Stress delta must be computed before updating vt's position
        ds = 0.0
        if self.structure_weight > 0:
            ds = self._stress_delta(vt, old_pos, new_pos)

        # Crossing delta via GLS incremental update
        prev_total = self.gls.compute_crossings()[0]
        self.gls.update_position(vt, new_pos)
        displacement = float(np.linalg.norm(self.gls.positions[vt] - old_pos))
        new_total = int(self.gls.compute_crossings()[0])
        delta_crossings = new_total - prev_total

        if self.soft_crossing:
            new_soft = self._compute_soft_crossings(self.coords)
            crossing_penalty = new_soft - self.current_soft_crossings
            self.current_soft_crossings = new_soft
        else:
            crossing_penalty = float(delta_crossings)

        reward = -crossing_penalty - self.structure_weight * ds

        self.current_stress += ds
        self.total_crossings = new_total

        # Renormalise layout if any node drifted outside [-1, 1]²
        # (crossings are scale/translation invariant, so total_crossings stays valid)
        if (self.coords > 1.0).any() or (self.coords < -1.0).any():
            lo, hi = self.coords.min(axis=0), self.coords.max(axis=0)
            scale = hi - lo
            scale[scale == 0] = 1.0
            self.gls.batch_update(
                range(self.num_nodes),
                2.0 * (self.coords - lo) / scale - 1.0,
            )

        if new_total < self.best_crossings:
            self.best_crossings = new_total
            self.best_coords = self.coords.copy()
            self.no_improve_steps = 0
        else:
            self.no_improve_steps += 1

        self.step_idx += 1
        terminated = self.total_crossings <= 0
        static = (self.config.min_effective_action > 0
                  and displacement < self.config.min_effective_action)
        patience_exceeded = (self.config.patience > 0
                             and self.no_improve_steps >= self.config.patience)

        if patience_exceeded and self.config.reset_to_best and not terminated:
            # Reset layout to best known position and continue the episode
            self.gls.batch_update(range(self.num_nodes), self.best_coords)
            self.total_crossings = self.best_crossings
            self.no_improve_steps = 0
            np.random.shuffle(self.node_order)
            if self.soft_crossing:
                self.current_soft_crossings = self._compute_soft_crossings(self.coords)
            self.current_stress = self._compute_structure_stress()
            patience_exceeded = False

        truncated = (self.step_idx >= self.config.max_steps
                     or static or patience_exceeded) and not terminated

        return self._build_obs(), reward, terminated, truncated, {
            "vt": vt,
            "step": self.step_idx,
            "delta_crossings": delta_crossings,
            "total_crossings": self.total_crossings,
            "displacement": displacement,
            "static_truncation": static,
            "crossings": self.total_crossings,
        }

    # ── Policy interface ───────────────────────────────────────────────────────

    def get_graph_data(self):
        edge_index_np = self._build_edge_index()  # [2, 2E] numpy
        edge_index = torch.tensor(edge_index_np, dtype=torch.long)
        src, dst = edge_index_np[0], edge_index_np[1]
        lengths = np.sqrt(
            ((self.coords[src] - self.coords[dst]) ** 2).sum(axis=-1, keepdims=True)
        ).astype(np.float32)
        edge_attr = torch.tensor(lengths, dtype=torch.float32)
        return {
            "node_features": self._build_obs(),
            "edge_index": edge_index,
            "edge_attr": edge_attr,
        }

    def get_coords(self) -> np.ndarray:
        return self.coords.copy()

    def get_policy_input(self, obs: np.ndarray, device) -> tuple:
        return (torch.tensor(obs, dtype=torch.float32, device=device), )

    @staticmethod
    def make_batch_input(envs, obs_list, device):
        return [
            torch.tensor(obs, dtype=torch.float32, device=device)
            for obs in obs_list
        ]
