# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Cython-accelerated global crossing detector — inverted logic.

Default: ALL edge pairs are crossing.
Pipeline removes pairs that can be *proven* non-crossing.

Compile with:
    poetry run python build.py build_ext --inplace
"""

import numpy as np
cimport numpy as np
from typing import Optional, Tuple


# ── Pre-computation ────────────────────────────────────────────────────────────

def precompute_geometry(
    positions: np.ndarray,
    edges: np.ndarray,
) -> dict:
    diff      = positions[:, None, :] - positions[None, :, :]
    node_dist = np.sqrt((diff * diff).sum(axis=-1))

    p1s = positions[edges[:, 0]]
    p2s = positions[edges[:, 1]]
    midpoints = (p1s + p2s) * 0.5
    half_lens = node_dist[edges[:, 0], edges[:, 1]] * 0.5

    mid_diff = midpoints[:, None, :] - midpoints[None, :, :]
    mid_dist = np.sqrt((mid_diff * mid_diff).sum(axis=-1))

    return {
        "node_dist": node_dist,
        "midpoints": midpoints,
        "half_lens": half_lens,
        "mid_dist":  mid_dist,
    }


# ── Main detector ──────────────────────────────────────────────────────────────

def find_all_crossings(
    positions,
    edges,
    precomputed=None,
    double node_eps  = 1e-4,
    double cross_eps = 1e-6,
    double degen_eps = 1e-3,
):
    cdef int E = len(edges)

    if E < 2:
        return 0, np.zeros((E, E), dtype=bool)

    if precomputed is None:
        precomputed = precompute_geometry(positions, edges)

    cdef np.ndarray[np.float64_t, ndim=2] nd_arr = np.ascontiguousarray(
        precomputed["node_dist"], dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] hl_arr = np.ascontiguousarray(
        precomputed["half_lens"],  dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] md_arr = np.ascontiguousarray(
        precomputed["mid_dist"],   dtype=np.float64)
    cdef np.ndarray[np.int64_t,   ndim=2] e_arr  = np.ascontiguousarray(
        edges, dtype=np.int64)

    cdef np.ndarray[np.uint8_t, ndim=2] mask_arr = np.zeros((E, E), dtype=np.uint8)

    # ── Initialise: all upper-triangle pairs are crossing ─────────────────
    ei_all, ej_all = np.triu_indices(E, k=1)
    mask_arr[ei_all, ej_all] = 1

    # ── Phase 1: Separate degenerate pairs ────────────────────────────────
    is_degen = (hl_arr[ei_all] < degen_eps) | (hl_arr[ej_all] < degen_eps)
    ei_nd = ei_all[~is_degen]
    ej_nd = ej_all[~is_degen]

    if len(ei_nd) == 0:
        result = mask_arr.view(np.bool_)
        return int(result.sum()), result

    # ── Phase 2: Midpoint distance filter — proven non-crossing ───────────
    reach   = hl_arr[ei_nd] + hl_arr[ej_nd]
    too_far = md_arr[ei_nd, ej_nd] > reach
    mask_arr[ei_nd[too_far], ej_nd[too_far]] = 0

    ei_c = ei_nd[~too_far]
    ej_c = ej_nd[~too_far]

    if len(ei_c) == 0:
        result = mask_arr.view(np.bool_)
        return int(result.sum()), result

    # ── Phase 3: Positional coincidence guard ─────────────────────────────
    u1 = e_arr[ei_c, 0];  v1 = e_arr[ei_c, 1]
    u2 = e_arr[ej_c, 0];  v2 = e_arr[ej_c, 1]

    pos_coincide = (
        ((u1 != u2) & (nd_arr[u1, u2] < node_eps)) |
        ((u1 != v2) & (nd_arr[u1, v2] < node_eps)) |
        ((v1 != u2) & (nd_arr[v1, u2] < node_eps)) |
        ((v1 != v2) & (nd_arr[v1, v2] < node_eps))
    )

    ei_t = ei_c[~pos_coincide]
    ej_t = ej_c[~pos_coincide]

    if len(ei_t) == 0:
        result = mask_arr.view(np.bool_)
        return int(result.sum()), result

    # ── Phase 4: Exact cross-product test — remove proven non-crossing ─────
    pos_c = np.ascontiguousarray(positions, dtype=np.float64)

    A = pos_c[e_arr[ei_t, 0]];  B = pos_c[e_arr[ei_t, 1]]
    C = pos_c[e_arr[ej_t, 0]];  D = pos_c[e_arr[ej_t, 1]]

    r   = B - A
    s   = D - C
    qmp = C - A

    rxs   = r[:, 0] * s[:, 1]   - r[:, 1] * s[:, 0]
    qmpxr = qmp[:, 0] * r[:, 1] - qmp[:, 1] * r[:, 0]
    qmpxs = qmp[:, 0] * s[:, 1] - qmp[:, 1] * s[:, 0]

    parallel = np.abs(rxs) < 1e-10
    safe_rxs = np.where(parallel, 1.0, rxs)

    t_p = qmpxs / safe_rxs
    u_p = qmpxr / safe_rxs

    proper = (
        ~parallel &
        (t_p > cross_eps) & (t_p < 1.0 - cross_eps) &
        (u_p > cross_eps) & (u_p < 1.0 - cross_eps)
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
    mask_arr[ei_t[not_crosses], ej_t[not_crosses]] = 0

    result = mask_arr.view(np.bool_)
    return int(result.sum()), result
