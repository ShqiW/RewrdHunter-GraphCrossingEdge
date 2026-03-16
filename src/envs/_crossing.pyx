# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Cython-compiled segment intersection batch check.

Replaces the pure-numpy _segments_intersect_batch in sequential.py.
Compile with:  poetry run python build.py build_ext --inplace
"""
import numpy as np
cimport numpy as np
from libc.math cimport fabs

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t


def segments_intersect_batch(
    const DTYPE_t[::1] p1,
    const DTYPE_t[::1] p2,
    const DTYPE_t[:, ::1] p3s,
    const DTYPE_t[:, ::1] p4s,
    double eps = 1e-6,
):
    """
    Check if segment (p1, p2) intersects each segment (p3s[i], p4s[i]).

    Phase 1 — AABB pre-filter: skip pairs whose bounding boxes don't overlap.
    Phase 2 — Exact cross-product test + collinear-overlap check.

    Returns bool array of shape (M,).
    Uses typed memoryviews so the inner loop releases the GIL.
    """
    cdef int m = p3s.shape[0]
    cdef np.ndarray[np.uint8_t, ndim=1] result_arr = np.zeros(m, dtype=np.uint8)
    cdef np.uint8_t[::1] result = result_arr   # memoryview for nogil access

    # Pre-compute p1/p2 derived values before releasing GIL
    cdef double rx = p2[0] - p1[0]
    cdef double ry = p2[1] - p1[1]
    cdef double r_sq = rx * rx + ry * ry
    cdef double safe_r_sq = r_sq if r_sq > 1e-10 else 1.0

    cdef double p1_xmin = p1[0] if p1[0] < p2[0] else p2[0]
    cdef double p1_xmax = p1[0] if p1[0] > p2[0] else p2[0]
    cdef double p1_ymin = p1[1] if p1[1] < p2[1] else p2[1]
    cdef double p1_ymax = p1[1] if p1[1] > p2[1] else p2[1]

    cdef double p3x, p3y, p4x, p4y
    cdef double x_min, x_max, y_min, y_max
    cdef double sx, sy, qmpx, qmpy
    cdef double rxs, qmpxr, qmpxs, t, u, t3, t4, t_lo, t_hi
    cdef int i

    with nogil:
        for i in range(m):
            p3x = p3s[i, 0]; p3y = p3s[i, 1]
            p4x = p4s[i, 0]; p4y = p4s[i, 1]

            # ── Phase 1: AABB reject ───────────────────────────────────────────
            x_min = p3x if p3x < p4x else p4x
            x_max = p3x if p3x > p4x else p4x
            y_min = p3y if p3y < p4y else p4y
            y_max = p3y if p3y > p4y else p4y

            if p1_xmax < x_min or p1_xmin > x_max or p1_ymax < y_min or p1_ymin > y_max:
                continue

            # ── Phase 2: exact cross-product test ─────────────────────────────
            sx   = p4x - p3x;  sy   = p4y - p3y
            qmpx = p3x - p1[0]; qmpy = p3y - p1[1]

            rxs   = rx * sy   - ry * sx
            qmpxr = qmpx * ry - qmpy * rx
            qmpxs = qmpx * sy - qmpy * sx

            if fabs(rxs) < 1e-10:
                # Parallel — check collinear overlap
                if fabs(qmpxr) < 1e-10:
                    t3  = (qmpx * rx + qmpy * ry) / safe_r_sq
                    t4  = t3 + (sx * rx + sy * ry) / safe_r_sq
                    t_lo = t3 if t3 < t4 else t4
                    t_hi = t3 if t3 > t4 else t4
                    if t_hi > eps and t_lo < 1.0 - eps and t_hi - t_lo > eps:
                        result[i] = 1
            else:
                t = qmpxs / rxs
                u = qmpxr / rxs
                if t > eps and t < 1.0 - eps and u > eps and u < 1.0 - eps:
                    result[i] = 1

    return result_arr.view(np.bool_)
