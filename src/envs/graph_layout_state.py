"""
GraphLayoutState — cached, incrementally-updated graph layout.

Wraps a fixed nx.Graph topology with mutable node coordinates and a
visibility mask (for sequential node-placement scenarios).

Cache maintained
---------------
  _node_dist  (N, N)  — Euclidean distances between all visible node pairs
                         (NaN for rows/cols of invisible nodes)
  _half_lens  (E,)    — half edge-length for active edges (both endpoints
                         visible); NaN for inactive edges
  _midpoints  (E, 2)  — midpoints of active edges; NaN for inactive edges
  _mid_dist   (E, E)  — pairwise midpoint distances for active edges;
                         NaN for inactive edges

Update strategy (lazy / dirty-tracking)
----------------------------------------
All write operations (update_position, unmask_node, batch_update,
apply_deltas) only write to self.positions and mark affected nodes as
dirty.  The cache is flushed at the start of compute_crossings() in
three vectorised passes:

  Pass 1 — node_dist: recompute rows/cols for all dirty visible nodes
  Pass 2 — half_lens / midpoints: recompute for all edges incident to
            any dirty node (reads from up-to-date node_dist)
  Pass 3 — mid_dist: recompute rows/cols for all edges affected in
            pass 2 (reads from up-to-date midpoints)

Cost of flushing k dirty nodes with total degree d:
  O(k·N  +  d·E)
"""

from __future__ import annotations

from typing import Collection, Iterable, Optional, Tuple
from numpy.typing import ArrayLike

import numpy as np
import networkx as nx

from src.envs._crossing_all import find_all_crossings


class GraphLayoutState:
    """
    Maintains node coordinates, visibility, and crossing-detection caches
    for a graph with fixed topology.

    Parameters
    ----------
    G               full graph (topology never changes after init)
    visible_nodes   nodes that are initially visible
    coords          (len(visible_nodes), 2) initial coordinates
    node_eps        threshold for treating two nodes as overlapping
    cross_eps       interior-only intersection margin (excludes endpoints)
    """

    def __init__(
        self,
        G: nx.Graph,
        visible_nodes: Collection,
        coords: np.ndarray,
        node_eps: float = 1e-4,
        cross_eps: float = 1e-6,
    ) -> None:
        self.node_eps  = node_eps
        self.cross_eps = cross_eps

        # ── Canonical node ordering ───────────────────────────────────────────
        self.node_list = list(G.nodes())
        self.N = len(self.node_list)
        self.n2i: dict = {n: i for i, n in enumerate(self.node_list)}

        # ── Edge arrays (fixed topology) ──────────────────────────────────────
        raw_edges = [(self.n2i[u], self.n2i[v]) for u, v in G.edges()]
        self.E = len(raw_edges)
        self.edges = (
            np.array(raw_edges, dtype=np.int64)
            if raw_edges
            else np.zeros((0, 2), dtype=np.int64)
        )

        # Node → list of incident edge indices (for incremental updates)
        self.n2e: list[list[int]] = [[] for _ in range(self.N)]
        for e_idx, (u, v) in enumerate(self.edges.tolist()):
            self.n2e[u].append(e_idx)
            self.n2e[v].append(e_idx)

        # ── Mutable state ─────────────────────────────────────────────────────
        self.positions = np.full((self.N, 2), np.nan, dtype=np.float64)
        self.visible   = np.zeros(self.N, dtype=bool)

        visible_list = list(visible_nodes)
        coords_arr   = np.asarray(coords, dtype=np.float64)
        if len(visible_list) != len(coords_arr):
            raise ValueError(
                f"visible_nodes length ({len(visible_list)}) must match "
                f"coords rows ({len(coords_arr)})"
            )
        for node, coord in zip(visible_list, coords_arr):
            idx = self.n2i[node]
            self.positions[idx] = coord
            self.visible[idx]   = True

        # ── Geometry caches (all NaN until rebuilt) ───────────────────────────
        self._node_dist = np.full((self.N, self.N), np.nan, dtype=np.float64)
        self._half_lens = np.full(self.E,            np.nan, dtype=np.float64)
        self._midpoints = np.full((self.E, 2),       np.nan, dtype=np.float64)
        self._mid_dist  = np.full((self.E, self.E),  np.nan, dtype=np.float64)

        self._dirty_nodes: set[int] = set()
        # Accumulated position deltas since the last flush (N, 2).
        # _delta_valid[i] is True only when node i was moved exclusively via
        # apply_deltas since the last flush; False after batch_update /
        # update_position (absolute write).  The distinction lets _flush_dirty
        # do incremental midpoint updates when both endpoints are delta-valid.
        self._node_deltas: np.ndarray = np.zeros((self.N, 2), dtype=np.float64)
        self._delta_valid: np.ndarray = np.zeros(self.N, dtype=bool)

        # Full rebuild for all initially visible nodes
        if visible_list:
            self._full_rebuild()

    # ══════════════════════════════════════════════════════════════════════════
    # Public write API
    # ══════════════════════════════════════════════════════════════════════════

    def update_position(self, node, coord: ArrayLike) -> None:
        """Move a single already-visible node to coord."""
        idx = self.n2i[node]
        if not self.visible[idx]:
            raise ValueError(f"Node {node!r} is not visible; use unmask_node() first.")
        coord_arr = np.asarray(coord, dtype=np.float64)
        self._apply_deltas(
            np.array([idx], dtype=np.int64),
            (coord_arr - self.positions[idx])[np.newaxis],
        )

    def unmask_node(self, node, coord: ArrayLike) -> None:
        """Reveal a previously invisible node at coord."""
        idx = self.n2i[node]
        if self.visible[idx]:
            raise ValueError(f"Node {node!r} is already visible.")
        # Initialise to zero so that _apply_deltas' "+= coord" lands at coord.
        # visible must be set before _apply_deltas so dirty-flush treats it as
        # a visible node.
        self.positions[idx] = 0.0
        self.visible[idx]   = True
        self._apply_deltas(
            np.array([idx], dtype=np.int64),
            np.asarray(coord, dtype=np.float64)[np.newaxis],
        )

    def batch_update(self, nodes: Iterable, coords: ArrayLike) -> None:
        """Set absolute positions for multiple visible nodes."""
        nodes_list = list(nodes)
        coords_arr = np.asarray(coords, dtype=np.float64)
        indices    = self._check_all_visible(nodes_list)
        idx_arr    = np.array(indices, dtype=np.int64)
        self._apply_deltas(idx_arr, coords_arr - self.positions[idx_arr])

    def apply_deltas(self, nodes: Iterable, deltas: ArrayLike) -> None:
        """Add (dx, dy) deltas to the coordinates of multiple visible nodes.

        Validates visibility for all nodes up-front (atomic), then delegates
        to ``_apply_deltas``.
        """
        nodes_list = list(nodes)
        deltas_arr = np.asarray(deltas, dtype=np.float64)
        indices    = self._check_all_visible(nodes_list)
        self._apply_deltas(np.array(indices, dtype=np.int64), deltas_arr)

    # ══════════════════════════════════════════════════════════════════════════
    # Crossing computation
    # ══════════════════════════════════════════════════════════════════════════

    def compute_crossings(self) -> Tuple[int, np.ndarray]:
        """
        Compute all crossing edge pairs among currently active edges.

        Flushes the dirty cache first, then delegates to find_all_crossings
        with the up-to-date precomputed geometry.

        Returns
        -------
        crossing_count : int
        crossing_mask  : (E, E) bool, upper-triangular, indexed by full edge
                         indices (self.edges[i] gives the node pair)
        """
        self._flush_dirty()

        # Active edges: both endpoints visible
        active_idx = self._active_edge_indices()
        E_active = len(active_idx)

        if E_active < 2:
            return 0, np.zeros((self.E, self.E), dtype=bool)

        active_edges = self.edges[active_idx]   # (E_active, 2)

        # Subset the precomputed dict for active edges only.
        # node_dist stays full (N×N) — Phase 0 needs it for overlap detection.
        sub_precomputed = {
            "node_dist": self._node_dist,
            "half_lens": self._half_lens[active_idx].copy(),
            "mid_dist":  self._mid_dist[np.ix_(active_idx, active_idx)].copy(),
        }

        count, sub_mask = find_all_crossings(
            self.positions,
            active_edges,
            precomputed=sub_precomputed,
            node_eps=self.node_eps,
            cross_eps=self.cross_eps,
        )

        # Expand sub_mask (E_active × E_active) back to full (E × E) indexing
        full_mask = np.zeros((self.E, self.E), dtype=bool)
        ai, aj = np.where(sub_mask)
        if len(ai):
            full_mask[active_idx[ai], active_idx[aj]] = True

        return count, full_mask

    # ══════════════════════════════════════════════════════════════════════════
    # Convenience read-only properties
    # ══════════════════════════════════════════════════════════════════════════

    @property
    def num_visible(self) -> int:
        return int(self.visible.sum())

    @property
    def visible_node_indices(self) -> np.ndarray:
        """Node indices (into self.node_list) that are currently visible."""
        return np.where(self.visible)[0]

    @property
    def visible_positions(self) -> np.ndarray:
        """(num_visible, 2) coordinates of visible nodes."""
        return self.positions[self.visible]

    def get_position(self, node) -> np.ndarray:
        """Coordinate of a single node (raises if invisible)."""
        idx = self.n2i[node]
        if not self.visible[idx]:
            raise ValueError(f"Node {node!r} is not visible.")
        return self.positions[idx].copy()

    # ══════════════════════════════════════════════════════════════════════════
    # Cache internals
    # ══════════════════════════════════════════════════════════════════════════

    def _apply_deltas(self, idx_arr: np.ndarray, deltas_arr: np.ndarray) -> None:
        """Core delta application — indices already resolved, no visibility check."""
        self.positions[idx_arr]    += deltas_arr
        self._node_deltas[idx_arr] += deltas_arr
        self._delta_valid[idx_arr]  = True
        self._dirty_nodes.update(idx_arr.tolist())

    def _check_all_visible(self, nodes: list) -> list[int]:
        """Return node indices after verifying every node is currently visible.

        Raises ``ValueError`` on the first invisible node so that callers can
        rely on an all-or-nothing guarantee before mutating any state.
        """
        indices: list[int] = []
        for node in nodes:
            idx = self.n2i[node]
            if not self.visible[idx]:
                raise ValueError(
                    f"Node {node!r} is not visible; use unmask_node() first."
                )
            indices.append(idx)
        return indices

    def _active_edge_indices(self) -> np.ndarray:
        """Indices of edges whose both endpoints are currently visible."""
        if self.E == 0:
            return np.array([], dtype=np.int64)
        u_vis = self.visible[self.edges[:, 0]]
        v_vis = self.visible[self.edges[:, 1]]
        return np.where(u_vis & v_vis)[0]

    def _full_rebuild(self) -> None:
        """Recompute the entire cache from scratch (used at init only)."""
        vis_idx = np.where(self.visible)[0]
        if len(vis_idx) == 0:
            return

        # node_dist for visible × visible
        vis_pos  = self.positions[vis_idx]                      # (V, 2)
        diff     = vis_pos[:, None, :] - vis_pos[None, :, :]   # (V, V, 2)
        vis_dist = np.sqrt((diff * diff).sum(axis=-1))          # (V, V)
        self._node_dist[np.ix_(vis_idx, vis_idx)] = vis_dist

        # midpoints and half_lens for active edges (vectorised)
        active_idx = self._active_edge_indices()
        if len(active_idx) > 0:
            u_act = self.edges[active_idx, 0]
            v_act = self.edges[active_idx, 1]
            self._midpoints[active_idx] = (
                self.positions[u_act] + self.positions[v_act]
            ) * 0.5
            self._half_lens[active_idx] = self._node_dist[u_act, v_act] * 0.5

            # mid_dist for active × active
            mids     = self._midpoints[active_idx]              # (E_act, 2)
            mid_diff = mids[:, None, :] - mids[None, :, :]     # (E_act, E_act, 2)
            mid_d    = np.sqrt((mid_diff * mid_diff).sum(axis=-1))
            self._mid_dist[np.ix_(active_idx, active_idx)] = mid_d

        self._dirty_nodes.clear()
        self._node_deltas.fill(0.0)
        self._delta_valid.fill(False)

    def _flush_dirty(self) -> None:
        """
        Apply all pending incremental cache updates in three vectorised passes.

        Pass 1: node_dist rows/cols for all k dirty visible nodes at once —
                one (k, V, 2) broadcast replaces k Python loop iterations.
        Pass 2: half_lens / midpoints for edges incident to dirty nodes,
                batched over all dirty edges simultaneously.
        Pass 3: mid_dist rows/cols for dirty-edge set vs. all active edges —
                one (D, E_act, 2) broadcast replaces D iterations.
        """
        if not self._dirty_nodes:
            return

        dirty     = [i for i in self._dirty_nodes if self.visible[i]]
        dirty_arr = np.array(dirty, dtype=np.int64)
        vis_idx   = np.where(self.visible)[0]

        # ── Pass 1: node distances (vectorised over k dirty nodes) ────────────
        # Reads : positions (always current)
        # Writes: _node_dist rows/cols for dirty visible nodes
        if len(dirty_arr) > 0 and len(vis_idx) > 0:
            vis_pos   = self.positions[vis_idx]                         # (V, 2)
            dirty_pos = self.positions[dirty_arr]                       # (k, 2)
            diff      = dirty_pos[:, None, :] - vis_pos[None, :, :]    # (k, V, 2)
            dists     = np.sqrt((diff * diff).sum(axis=-1))             # (k, V)
            self._node_dist[np.ix_(dirty_arr, vis_idx)] = dists
            self._node_dist[np.ix_(vis_idx, dirty_arr)] = dists.T

        # ── Collect dirty edges ───────────────────────────────────────────────
        dirty_edges_set: set[int] = set()
        for idx in dirty:
            dirty_edges_set.update(self.n2e[idx])

        if dirty_edges_set:
            de_arr        = np.array(sorted(dirty_edges_set), dtype=np.int64)
            u_de          = self.edges[de_arr, 0]
            v_de          = self.edges[de_arr, 1]
            both_vis_mask = self.visible[u_de] & self.visible[v_de]

            # ── Pass 2: midpoints and half_lens ──────────────────────────────
            # Reads : positions, _node_dist (updated in pass 1)
            # Writes: _midpoints, _half_lens for all dirty edges
            #
            # Inactive dirty edges → NaN
            inactive = de_arr[~both_vis_mask]
            if len(inactive):
                self._midpoints[inactive] = np.nan
                self._half_lens[inactive] = np.nan

            # Active dirty edges
            ae   = de_arr[both_vis_mask]
            u_ae = u_de[both_vis_mask]
            v_ae = v_de[both_vis_mask]
            if len(ae):
                dv_both  = self._delta_valid[u_ae] & self._delta_valid[v_ae]
                mp_valid = ~np.isnan(self._midpoints[ae, 0])
                inc      = dv_both & mp_valid   # can use incremental update

                # Incremental: midpoint += (Δu + Δv) / 2
                if inc.any():
                    ie = ae[inc]
                    self._midpoints[ie] += (
                        self._node_deltas[u_ae[inc]] + self._node_deltas[v_ae[inc]]
                    ) * 0.5

                # Absolute recompute for non-incremental active edges
                scratch = ~inc
                if scratch.any():
                    se = ae[scratch]
                    self._midpoints[se] = (
                        self.positions[u_ae[scratch]] + self.positions[v_ae[scratch]]
                    ) * 0.5

                # half_lens always reads updated node_dist from pass 1
                self._half_lens[ae] = self._node_dist[u_ae, v_ae] * 0.5

            # ── Pass 3: mid_dist rows/cols (vectorised over D dirty edges) ────
            # Reads : _midpoints (updated in pass 2)
            # Writes: _mid_dist rows/cols for dirty edges vs. all active edges
            active_idx = self._active_edge_indices()
            if len(active_idx) > 0:
                active_mids = self._midpoints[active_idx]   # (E_act, 2)
                nan_mask    = np.isnan(self._half_lens[de_arr])

                # Deactivated dirty edges: clear row/col
                deact = de_arr[nan_mask]
                if len(deact):
                    self._mid_dist[deact, :] = np.nan
                    self._mid_dist[:, deact] = np.nan

                # Active dirty edges: one broadcast for all D vs E_act pairs
                act_de = de_arr[~nan_mask]
                if len(act_de):
                    dirty_mids = self._midpoints[act_de]                          # (D, 2)
                    diff_md    = dirty_mids[:, None, :] - active_mids[None, :, :] # (D, E_act, 2)
                    dists_md   = np.sqrt((diff_md * diff_md).sum(axis=-1))        # (D, E_act)
                    self._mid_dist[np.ix_(act_de, active_idx)] = dists_md
                    self._mid_dist[np.ix_(active_idx, act_de)] = dists_md.T

        self._dirty_nodes.clear()
        # Reset accumulated deltas for dirty nodes so the next update cycle
        # does not double-count old deltas in the incremental midpoint formula.
        if len(dirty_arr):
            self._node_deltas[dirty_arr] = 0.0
            self._delta_valid[dirty_arr] = False
