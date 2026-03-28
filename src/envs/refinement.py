"""
Iterative layout refinement environment for graph layout optimization.

Starts with a full layout and moves nodes to reduce crossings and stress.
Action: continuous 2D delta offset (dx, dy)
Reward: -new_crossings(v_t) + λ·ΔStructure
Termination: Fixed max_steps.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import networkx as nx
import torch
from gymnasium import spaces

from src.tasks.base import BaseEnvConfig
from src.envs.base import BaseGraphEnv
from src.data.rome import GraphData
from src.envs.utils import get_initial_layout
from src.envs.sequential import _segments_intersect_batch_dispatch


@dataclass
class RefinementEnvConfig(BaseEnvConfig):
    """Refinement environment configuration"""
    structure_weight: float = 0.3
    exclude_non_crossing: bool = False  # focus only on nodes involved in crossings
    order_method: str = "bfs"
    delta_scale: float = 0.1  # max offset per step in [-1,1]² space


class RefinementGraphEnv(BaseGraphEnv):
    """
    Iterative refinement environment.
    Cycles through nodes (optionally only those in crossings) and applies local moves.
    """

    def __init__(
        self,
        graph_data: GraphData,
        device,
        config: RefinementEnvConfig,
    ):
        super().__init__()

        self.config = config
        self.device = device
        self.structure_weight = config.structure_weight
        self.delta_scale = config.delta_scale

        self.neato_coords = graph_data.neato_coords.numpy()
        self.num_nodes = graph_data.num_nodes
        self.graph_name = graph_data.graph_name

        edge_index_np = graph_data.edge_index.numpy()
        self.adj = [[] for _ in range(self.num_nodes)]
        self.undirected_edges = []
        seen = set()
        for i in range(edge_index_np.shape[1]):
            u, v = int(edge_index_np[0, i]), int(edge_index_np[1, i])
            if u < v and (u, v) not in seen:
                seen.add((u, v))
                self.undirected_edges.append((u, v))
                self.adj[u].append(v)
                self.adj[v].append(u)

        self.num_edges = len(self.undirected_edges)
        self.graph_distance = graph_data.graph_distance.numpy().astype(
            np.float32)
        self.tau = graph_data.tau

        # ── Precomputed fixed arrays (graph topology never changes) ────────────
        if self.undirected_edges:
            self._edge_arr = np.array(self.undirected_edges, dtype=np.int32)
            self._eu = self._edge_arr[:, 0]
            self._ev = self._edge_arr[:, 1]
        else:
            self._edge_arr = np.zeros((0, 2), dtype=np.int32)
            self._eu = np.zeros(0, dtype=np.int32)
            self._ev = np.zeros(0, dtype=np.int32)

        # Stress precomputation: row/col indices + weights, all fixed by graph topology
        k = self.num_nodes
        triu = np.triu(np.ones((k, k), dtype=bool), k=1)
        stress_mask = triu & (self.graph_distance > 0)
        self._stress_rows, self._stress_cols = np.where(stress_mask)
        d_g_masked = self.graph_distance[self._stress_rows, self._stress_cols]
        self._stress_W = 1.0 / np.maximum(d_g_masked**2, 1e-3)
        self._stress_d_g = d_g_masked
        self._stress_norm = float(k * (k - 1) / 2) if k > 1 else 1.0

        # Per-node incremental stress: O(N) per step instead of O(N²)
        # For node vt moving, only its distances to all others change.
        node_mask = ~np.eye(k, dtype=bool) & (self.graph_distance > 0)
        self._stress_others = [np.where(node_mask[vt])[0] for vt in range(k)]
        self._stress_d_g_from = [
            self.graph_distance[vt, self._stress_others[vt]] for vt in range(k)
        ]
        self._stress_W_from = [
            1.0 / np.maximum(d**2, 1e-3) for d in self._stress_d_g_from
        ]

        # Crossing mask precomputation: per-(vt, neighbor) valid edge index arrays
        eu, ev = self._eu, self._ev
        self._adj_valid_idx: list = []
        for vt in range(self.num_nodes):
            nb_indices = []
            for nb in self.adj[vt]:
                valid = (eu != nb) & (ev != nb) & (eu != vt) & (ev != vt)
                nb_indices.append(np.where(valid)[0])
            self._adj_valid_idx.append(nb_indices)

        self._nx_graph = nx.Graph()
        self._nx_graph.add_nodes_from(range(self.num_nodes))
        self._nx_graph.add_edges_from(self.undirected_edges)

        self.action_space = spaces.Box(low=-1.0,
                                       high=1.0,
                                       shape=(2, ),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(self.num_nodes, 4),
                                            dtype=np.float32)

        # Episode state
        self.coords: np.ndarray = np.zeros((self.num_nodes, 2),
                                           dtype=np.float32)
        self.node_order: list = []
        self.step_idx: int = 0
        self.current_stress: float = 0.0
        self.total_crossings: int = 0

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
        if not self.undirected_edges: return []
        eu, ev = self._eu, self._ev
        p3s, p4s = self.coords[eu], self.coords[ev]
        nodes = set()
        for i in range(len(self._edge_arr)):
            p1, p2 = p3s[i], p4s[i]
            u, v = int(eu[i]), int(ev[i])
            mask = (eu != u) & (eu != v) & (ev != u) & (ev != v)
            if not mask.any(): continue
            hits = _segments_intersect_batch_dispatch(p1, p2, p3s[mask],
                                                      p4s[mask])
            if hits.any():
                nodes.add(u)
                nodes.add(v)
        return sorted(list(nodes))

    def _compute_node_order(self) -> list:
        # For simplicity, default to BFS or random
        n = self.num_nodes
        order = list(range(n))
        np.random.shuffle(order)
        return order

    def _count_total_crossings(self) -> int:
        if len(self.undirected_edges) < 2:
            return 0
        eu, ev = self._eu, self._ev
        p3s, p4s = self.coords[eu], self.coords[ev]
        total = 0
        for i in range(len(self._edge_arr) - 1):
            u, v = int(eu[i]), int(ev[i])
            mask = (eu[i + 1:] != u) & (eu[i + 1:] != v) & (
                ev[i + 1:] != u) & (ev[i + 1:] != v)
            if not mask.any():
                continue
            hits = _segments_intersect_batch_dispatch(p3s[i], p4s[i],
                                                      p3s[i + 1:][mask],
                                                      p4s[i + 1:][mask])
            total += int(hits.sum())
        return total

    def _count_crossings_for_node(
        self,
        vt: int,
        pos: np.ndarray,
        all_p3: np.ndarray,
        all_p4: np.ndarray,
    ) -> int:
        total = 0
        for i, nb in enumerate(self.adj[vt]):
            idx = self._adj_valid_idx[vt][i]
            if len(idx) == 0: continue
            if np.linalg.norm(pos - self.coords[nb]) < 1e-6:
                total += len(idx)
                continue
            hits = _segments_intersect_batch_dispatch(pos, self.coords[nb],
                                                      all_p3[idx], all_p4[idx])
            total += int(hits.sum())
        return total

    def _compute_structure_stress(self) -> float:
        """Full O(N²) stress — used only at reset()."""
        if self.num_nodes < 2 or len(self._stress_rows) == 0: return 0.0
        d_l = np.sqrt(((self.coords[self._stress_rows] -
                        self.coords[self._stress_cols])**2).sum(axis=-1) +
                      1e-6)
        stress = (self._stress_W * (d_l - self._stress_d_g)**2).sum()
        return float(stress / self._stress_norm)

    def _stress_delta(self, vt: int, old_pos: np.ndarray,
                      new_pos: np.ndarray) -> float:
        """Incremental O(N) stress change when moving vt from old_pos to new_pos."""
        others = self._stress_others[vt]
        if len(others) == 0: return 0.0
        coords_others = self.coords[others]  # [N-1, 2]  (vt not yet updated)
        d_g = self._stress_d_g_from[vt]  # [N-1]
        W = self._stress_W_from[vt]  # [N-1]
        d_old = np.sqrt(((old_pos - coords_others)**2).sum(axis=-1) + 1e-6)
        d_new = np.sqrt(((new_pos - coords_others)**2).sum(axis=-1) + 1e-6)
        return float((W * ((d_new - d_g)**2 - (d_old - d_g)**2)).sum() /
                     self._stress_norm)

    def _build_obs(self):
        vt = self.node_order[self.step_idx % len(self.node_order)]
        neighbors = set(self.adj[vt])
        features = np.zeros((self.num_nodes, 4), dtype=np.float32)
        features[:, 0:2] = self.coords
        features[vt, 2] = 1.0
        for nb in neighbors:
            features[nb, 3] = 1.0
        return features

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.coords = self._get_initial_layout()
        self.node_order = self._compute_node_order()
        if self.config.exclude_non_crossing:
            crossing_nodes = self._get_nodes_in_crossings()
            if crossing_nodes:
                self.node_order = [
                    v for v in self.node_order if v in crossing_nodes
                ]

        self.step_idx = 0
        self.current_stress = self._compute_structure_stress()
        self.total_crossings = self._count_total_crossings()
        obs = self._build_obs()
        return obs, {
            "vt": self.node_order[0] if self.node_order else None,
            "step": 0,
            "total_crossings": self.total_crossings
        }

    def step(self, action: np.ndarray):
        if not self.node_order or self.step_idx >= self.config.max_steps:
            return self._build_obs(), 0.0, False, True, {
                "step": self.step_idx,
                "total_crossings": self.total_crossings
            }

        vt = self.node_order[self.step_idx % len(self.node_order)]
        old_pos = self.coords[vt].copy()
        pos = (old_pos + action * self.delta_scale).astype(np.float32)

        # Compute edge endpoints once — shared by both old and new crossing checks
        all_p3 = self.coords[self._eu]
        all_p4 = self.coords[self._ev]
        old_crossings = self._count_crossings_for_node(vt, old_pos, all_p3,
                                                       all_p4)
        new_crossings = self._count_crossings_for_node(vt, pos, all_p3, all_p4)
        delta_crossings = new_crossings - old_crossings
        reward = -float(delta_crossings)

        # Stress delta must be computed before coords[vt] is updated
        # (coords[others] is unchanged since only vt moves)
        if self.structure_weight > 0:
            ds = self._stress_delta(vt, old_pos, pos)
            reward -= self.structure_weight * ds
            self.current_stress += ds

        self.coords[vt] = pos
        self.total_crossings += delta_crossings

        # If any node is outside [-1, 1]², renormalize the whole layout.
        # Crossings are scale/translation invariant so total_crossings stays valid.
        if (self.coords > 1.0).any() or (self.coords < -1.0).any():
            lo, hi = self.coords.min(axis=0), self.coords.max(axis=0)
            scale = hi - lo
            scale[scale == 0] = 1.0
            self.coords = (2.0 * (self.coords - lo) / scale - 1.0).astype(
                np.float32)

        self.step_idx += 1
        terminated = (self.total_crossings <= 0)
        truncated = (self.step_idx >= self.config.max_steps) and not terminated
        obs = self._build_obs()
        info = {
            "vt": vt,
            "step": self.step_idx,
            "new_crossings": new_crossings,
            "total_crossings": self.total_crossings
        }
        return obs, reward, terminated, truncated, info

    def get_graph_data(self):
        return {
            "node_features": self._build_obs(),
            "edge_index": self._build_edge_index()
        }

    def _build_edge_index(self):
        src, dst = [], []
        for (u, v) in self.undirected_edges:
            src += [u, v]
            dst += [v, u]
        return np.array([src, dst], dtype=np.int64) if src else np.zeros(
            (2, 0), dtype=np.int64)

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
