"""
Numba JIT-compiled segment intersection batch check.

Used as a fallback when the Cython extension is not compiled.
First call triggers JIT compilation (~0.5s); subsequent calls are fast.
"""
import numpy as np
import numba


@numba.njit(cache=True, fastmath=True)
def segments_intersect_batch(
    p1: np.ndarray,
    p2: np.ndarray,
    p3s: np.ndarray,
    p4s: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Check if segment (p1, p2) intersects each segment (p3s[i], p4s[i]).
    AABB pre-filter + exact cross-product test + collinear-overlap check.
    Returns bool array of shape (M,).
    """
    m = p3s.shape[0]
    result = np.zeros(m, dtype=numba.boolean)

    rx = p2[0] - p1[0]
    ry = p2[1] - p1[1]
    r_sq = rx * rx + ry * ry
    safe_r_sq = r_sq if r_sq > 1e-10 else 1.0

    p1_xmin = p1[0] if p1[0] < p2[0] else p2[0]
    p1_xmax = p1[0] if p1[0] > p2[0] else p2[0]
    p1_ymin = p1[1] if p1[1] < p2[1] else p2[1]
    p1_ymax = p1[1] if p1[1] > p2[1] else p2[1]

    for i in range(m):
        p3x = p3s[i, 0]; p3y = p3s[i, 1]
        p4x = p4s[i, 0]; p4y = p4s[i, 1]

        # ── Phase 1: AABB reject ──────────────────────────────────────────
        x_min = p3x if p3x < p4x else p4x
        x_max = p3x if p3x > p4x else p4x
        y_min = p3y if p3y < p4y else p4y
        y_max = p3y if p3y > p4y else p4y

        if p1_xmax < x_min or p1_xmin > x_max or p1_ymax < y_min or p1_ymin > y_max:
            continue

        # ── Phase 2: cross-product test ───────────────────────────────────
        sx   = p4x - p3x;   sy   = p4y - p3y
        qmpx = p3x - p1[0]; qmpy = p3y - p1[1]

        rxs   = rx * sy   - ry * sx
        qmpxr = qmpx * ry - qmpy * rx
        qmpxs = qmpx * sy - qmpy * sx

        if abs(rxs) < 1e-10:
            if abs(qmpxr) < 1e-10:  # collinear
                t3 = (qmpx * rx + qmpy * ry) / safe_r_sq
                t4 = t3 + (sx * rx + sy * ry) / safe_r_sq
                t_lo = t3 if t3 < t4 else t4
                t_hi = t3 if t3 > t4 else t4
                if t_hi > eps and t_lo < 1.0 - eps and t_hi - t_lo > eps:
                    result[i] = True
        else:
            t = qmpxs / rxs
            u = qmpxr / rxs
            if eps < t < 1.0 - eps and eps < u < 1.0 - eps:
                result[i] = True

    return result
