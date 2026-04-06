"""
Global crossing edge detector — inverted logic.

Default: ALL edge pairs are crossing.
Pipeline removes pairs that can be *proven* non-crossing.

Pre-computation (can be cached and reused across calls):
  node_dist  (N, N) — pairwise node distances
  midpoints  (E, 2) — edge midpoints
  half_lens  (E,)   — edge half-lengths
  mid_dist   (E, E) — pairwise midpoint distances (coarse filter)

Pipeline
--------
Phase 1 — Degenerate edge separation
    Edges with half_len < degen_eps are degenerate (nodes collapsed).
    Degenerate edge pairs cannot be proven non-crossing — they stay True.
    Only non-degenerate pairs proceed to Phase 2 and 3.

Phase 2 — Midpoint distance filter (proven non-crossing)
    Remove pair (e1, e2) if mid_dist > h1 + h2 — geometrically impossible
    to intersect. Applied only to non-degenerate pairs.

Phase 3 — Positional coincidence guard
    If two DIFFERENT nodes (different indices) are at the same position
    (node_dist < node_eps), their incident edge pairs cannot be proven
    non-crossing — they stay True.

Phase 4 — Exact cross-product test
    For remaining candidates: run parametric segment intersection.
    Remove only pairs proven non-crossing.
"""
from __future__ import annotations

import numpy as np
from typing import Optional, Tuple


# ── Pre-computation ────────────────────────────────────────────────────────────

def precompute_geometry(
    positions: np.ndarray,  # (N, 2)
    edges: np.ndarray,      # (E, 2) int — node indices
) -> dict:
    """
    Compute all geometric quantities needed for crossing detection.

    Returns a dict with:
      node_dist : (N, N) float — pairwise Euclidean distances between nodes
      midpoints : (E, 2) float — edge midpoints
      half_lens : (E,)  float — half edge-lengths
      mid_dist  : (E, E) float — pairwise midpoint distances
    """
    diff = positions[:, None, :] - positions[None, :, :]   # (N, N, 2)
    node_dist = np.sqrt((diff * diff).sum(axis=-1))         # (N, N)

    p1s = positions[edges[:, 0]]                            # (E, 2)
    p2s = positions[edges[:, 1]]                            # (E, 2)
    midpoints = (p1s + p2s) * 0.5                           # (E, 2)
    half_lens = node_dist[edges[:, 0], edges[:, 1]] * 0.5  # (E,)

    mid_diff = midpoints[:, None, :] - midpoints[None, :, :]  # (E, E, 2)
    mid_dist = np.sqrt((mid_diff * mid_diff).sum(axis=-1))     # (E, E)

    return {
        "node_dist": node_dist,
        "midpoints": midpoints,
        "half_lens": half_lens,
        "mid_dist":  mid_dist,
    }


# ── Main detector ──────────────────────────────────────────────────────────────

def find_all_crossings(
    positions: np.ndarray,              # (N, 2)
    edges: np.ndarray,                  # (E, 2) int
    precomputed: Optional[dict] = None,
    node_eps: float = 1e-4,
    cross_eps: float = 1e-6,
    degen_eps: float = 1e-3,
) -> Tuple[int, np.ndarray]:
    """
    Find all crossing edge pairs using inverted logic:
    start with all pairs crossing, remove only those proven non-crossing.

    Parameters
    ----------
    positions   : (N, 2) node coordinates
    edges       : (E, 2) edge index pairs
    precomputed : cached dict from precompute_geometry()
    node_eps    : nodes closer than this are considered positionally coincident
    cross_eps   : intersection parameter must be in (cross_eps, 1-cross_eps)
    degen_eps   : edges with half_len < degen_eps are degenerate (collapsed);
                  they are never removed from the crossing mask

    Returns
    -------
    crossing_count : int
    crossing_mask  : (E, E) bool — upper-triangular; True where pair crosses
    """
    E = len(edges)

    if E < 2:
        return 0, np.zeros((E, E), dtype=bool)

    if precomputed is None:
        precomputed = precompute_geometry(positions, edges)

    node_dist = precomputed["node_dist"]  # (N, N)
    half_lens = precomputed["half_lens"]  # (E,)
    mid_dist  = precomputed["mid_dist"]   # (E, E)

    # ── Initialise: all upper-triangle pairs are crossing ─────────────────────
    ei_all, ej_all = np.triu_indices(E, k=1)
    crossing_mask = np.zeros((E, E), dtype=bool)
    crossing_mask[ei_all, ej_all] = True

    # ── Phase 1: Separate degenerate pairs ────────────────────────────────────
    # Degenerate edges have undefined direction — cannot be proven non-crossing.
    # They remain True throughout.  Only non-degenerate pairs proceed.
    is_degen = (half_lens[ei_all] < degen_eps) | (half_lens[ej_all] < degen_eps)
    ei_nd = ei_all[~is_degen]
    ej_nd = ej_all[~is_degen]

    if len(ei_nd) == 0:
        return int(crossing_mask.sum()), crossing_mask

    # ── Phase 2: Midpoint distance filter — proven non-crossing ───────────────
    reach   = half_lens[ei_nd] + half_lens[ej_nd]
    too_far = mid_dist[ei_nd, ej_nd] > reach
    crossing_mask[ei_nd[too_far], ej_nd[too_far]] = False

    ei_c = ei_nd[~too_far]
    ej_c = ej_nd[~too_far]

    if len(ei_c) == 0:
        return int(crossing_mask.sum()), crossing_mask

    # ── Phase 3: Positional coincidence guard ─────────────────────────────────
    # If any endpoint pair consists of DIFFERENT node indices at the same
    # position, the edges share that location — cannot be proven non-crossing.
    u1 = edges[ei_c, 0];  v1 = edges[ei_c, 1]
    u2 = edges[ej_c, 0];  v2 = edges[ej_c, 1]

    pos_coincide = (
        ((u1 != u2) & (node_dist[u1, u2] < node_eps)) |
        ((u1 != v2) & (node_dist[u1, v2] < node_eps)) |
        ((v1 != u2) & (node_dist[v1, u2] < node_eps)) |
        ((v1 != v2) & (node_dist[v1, v2] < node_eps))
    )

    ei_t = ei_c[~pos_coincide]
    ej_t = ej_c[~pos_coincide]

    if len(ei_t) == 0:
        return int(crossing_mask.sum()), crossing_mask

    # ── Phase 4: Exact cross-product test — remove proven non-crossing ────────
    A = positions[edges[ei_t, 0]]
    B = positions[edges[ei_t, 1]]
    C = positions[edges[ej_t, 0]]
    D = positions[edges[ej_t, 1]]

    r   = B - A
    s   = D - C
    qmp = C - A

    rxs   = r[:, 0] * s[:, 1]   - r[:, 1] * s[:, 0]
    qmpxr = qmp[:, 0] * r[:, 1] - qmp[:, 1] * r[:, 0]
    qmpxs = qmp[:, 0] * s[:, 1] - qmp[:, 1] * s[:, 0]

    parallel = np.abs(rxs) < 1e-10
    safe_rxs = np.where(parallel, 1.0, rxs)

    t = qmpxs / safe_rxs
    u = qmpxr / safe_rxs

    proper = (
        ~parallel &
        (t > cross_eps) & (t < 1.0 - cross_eps) &
        (u > cross_eps) & (u < 1.0 - cross_eps)
    )

    r_sq      = (r * r).sum(axis=1)
    safe_r_sq = np.where(r_sq > 1e-10, r_sq, 1.0)
    t3 = (qmp * r).sum(axis=1) / safe_r_sq
    t4 = t3 + (s * r).sum(axis=1) / safe_r_sq
    t_lo = np.minimum(t3, t4)
    t_hi = np.maximum(t3, t4)

    collinear_overlap = (
        parallel &
        (np.abs(qmpxr) < 1e-10) &
        (t_hi > cross_eps) &
        (t_lo < 1.0 - cross_eps) &
        (t_hi - t_lo > cross_eps)
    )

    not_crosses = ~proper & ~collinear_overlap
    crossing_mask[ei_t[not_crosses], ej_t[not_crosses]] = False

    return int(crossing_mask.sum()), crossing_mask
