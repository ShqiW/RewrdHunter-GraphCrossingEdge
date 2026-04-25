"""
Tests for GraphLayoutState — cache correctness and crossing detection.

Test strategy
-------------
Each test case uses a geometrically explicit graph (with a known number of crossing edges),
and verifies cache correctness via two methods:
  1. Compare the result of compute_crossings() against the expected value.
  2. Compare the result after incremental updates against the result after a full rebuild (consistency check).
"""
import copy
import time
import json
import os
import numpy as np
import networkx as nx
import pytest

from src.envs.graph_layout_state import GraphLayoutState
from src.envs._crossing_all import precompute_geometry, find_all_crossings


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_x_graph():
    """
    X-shaped graph: two crossing diagonals
    Nodes: 0=(0,0), 1=(1,1), 2=(0,1), 3=(1,0)
    Edges: 0-1, 2-3  -> intersect exactly at (0.5,0.5), crossing_count = 1
    """
    G = nx.Graph()
    G.add_edges_from([(0, 1), (2, 3)])
    coords = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    return G, list(G.nodes()), coords


def make_square_graph():
    """
    Square graph: four edges, no crossings
    Nodes: 0=(0,0), 1=(1,0), 2=(1,1), 3=(0,1)
    Edges: 0-1, 1-2, 2-3, 3-0
    """
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    return G, list(G.nodes()), coords


def make_k4_graph():
    """
    K4 complete graph (4 nodes) with a planar embedding -> 0 crossings
    Nodes: 0=(0,0), 1=(2,0), 2=(1,2), 3=(1,0.5) (center point)
    Planar graph, K4 can be embedded in the plane; known crossing-free coordinates used here.
    """
    G = nx.complete_graph(4)
    # Embedded inside a square with known coordinates that produce crossings
    coords = np.array([
        [0.0, 0.0],  # 0
        [2.0, 0.0],  # 1
        [2.0, 2.0],  # 2
        [0.0, 2.0],  # 3
    ])
    # K4 in this layout has crossings (diagonals 0-2 and 1-3 intersect), which is what we are testing
    return G, list(G.nodes()), coords


def reference_crossing_count(state: GraphLayoutState) -> int:
    """Compute crossing count from scratch using precompute_geometry (bypasses cache)."""
    active_idx = state._active_edge_indices()
    if len(active_idx) < 2:
        return 0
    active_edges = state.edges[active_idx]
    geo = precompute_geometry(state.positions, active_edges)
    count, _ = find_all_crossings(
        state.positions, active_edges, precomputed=geo,
        node_eps=state.node_eps, cross_eps=state.cross_eps,
    )
    return count


def assert_cache_consistent(state: GraphLayoutState):
    """
    Verify that the incremental cache matches the result computed from scratch.
    Also checks internal consistency of _node_dist, _half_lens, _midpoints, and _mid_dist.
    """
    # First flush (trigger incremental update)
    count_inc, mask_inc = state.compute_crossings()
    count_ref = reference_crossing_count(state)
    assert count_inc == count_ref, (
        f"Crossing count mismatch: cache={count_inc}, reference={count_ref}"
    )

    # Verify symmetry of _node_dist for visible nodes
    vis = np.where(state.visible)[0]
    d = state._node_dist[np.ix_(vis, vis)]
    assert np.allclose(d, d.T, atol=1e-10), "_node_dist is not symmetric"
    assert np.allclose(np.diag(d), 0.0, atol=1e-10), "_node_dist diagonal is nonzero"

    # Verify _half_lens matches actual coordinates
    active_idx = state._active_edge_indices()
    for e_idx in active_idx:
        u, v = state.edges[e_idx]
        expected_half = np.linalg.norm(state.positions[u] - state.positions[v]) / 2
        assert abs(state._half_lens[e_idx] - expected_half) < 1e-9, (
            f"edge {e_idx}: half_len cache={state._half_lens[e_idx]:.6f}, "
            f"expected={expected_half:.6f}"
        )
        expected_mid = (state.positions[u] + state.positions[v]) / 2
        assert np.allclose(state._midpoints[e_idx], expected_mid, atol=1e-9), (
            f"edge {e_idx}: midpoint cache error"
        )


# ── Initialization tests ──────────────────────────────────────────────────────

class TestInit:
    def test_basic_init(self):
        G, nodes, coords = make_x_graph()
        state = GraphLayoutState(G, nodes, coords)
        assert state.N == 4
        assert state.E == 2
        assert state.num_visible == 4

    def test_positions_stored(self):
        G, nodes, coords = make_x_graph()
        state = GraphLayoutState(G, nodes, coords)
        for node, coord in zip(nodes, coords):
            np.testing.assert_array_equal(state.get_position(node), coord)

    def test_partial_visible(self):
        """Initialization should work correctly when only a subset of nodes are visible."""
        G, nodes, coords = make_x_graph()
        state = GraphLayoutState(G, nodes[:2], coords[:2])
        assert state.num_visible == 2

    def test_coord_node_length_mismatch_raises(self):
        G, nodes, coords = make_x_graph()
        with pytest.raises(ValueError):
            GraphLayoutState(G, nodes, coords[:2])

    def test_empty_graph(self):
        G = nx.Graph()
        G.add_nodes_from([0, 1, 2])
        coords = np.zeros((3, 2))
        state = GraphLayoutState(G, list(G.nodes()), coords)
        count, mask = state.compute_crossings()
        assert count == 0
        assert mask.shape == (0, 0)

    def test_no_dirty_after_init(self):
        G, nodes, coords = make_x_graph()
        state = GraphLayoutState(G, nodes, coords)
        assert len(state._dirty_nodes) == 0


# ── Crossing detection correctness ────────────────────────────────────────────

class TestCrossings:
    def test_x_shape_has_one_crossing(self):
        G, nodes, coords = make_x_graph()
        state = GraphLayoutState(G, nodes, coords)
        count, mask = state.compute_crossings()
        assert count == 1
        assert mask.sum() == 1

    def test_square_has_no_crossing(self):
        G, nodes, coords = make_square_graph()
        state = GraphLayoutState(G, nodes, coords)
        count, _ = state.compute_crossings()
        assert count == 0

    def test_k4_corner_layout_crossings(self):
        """K4 in a square corner layout has 1 crossing (diagonals 0-2 and 1-3)."""
        G, nodes, coords = make_k4_graph()
        state = GraphLayoutState(G, nodes, coords)
        count, _ = state.compute_crossings()
        assert_cache_consistent(state)
        # Square K4: only diagonals {0,2} and {1,3} intersect
        assert count == 1

    def test_crossing_mask_upper_triangular(self):
        G, nodes, coords = make_x_graph()
        state = GraphLayoutState(G, nodes, coords)
        _, mask = state.compute_crossings()
        # Lower triangle must be all False
        lower = np.tril(mask, k=-1)
        assert not lower.any()

    def test_adjacent_edges_not_counted(self):
        """Edges sharing an endpoint should not be counted as crossing."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2)])
        coords = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 0.0]])
        state = GraphLayoutState(G, list(G.nodes()), coords)
        count, _ = state.compute_crossings()
        assert count == 0

    def test_parallel_non_overlapping_no_crossing(self):
        """Parallel but non-overlapping edges should not be judged as crossing."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (2, 3)])
        coords = np.array([[0.0, 0.0], [1.0, 0.0],
                           [0.0, 1.0], [1.0, 1.0]])
        state = GraphLayoutState(G, list(G.nodes()), coords)
        count, _ = state.compute_crossings()
        assert count == 0

    def test_two_diagonal_crossings(self):
        """
        Two diagonals (a variant of X-shape using make_x_graph nodes but different edges), confirmed to have 1 crossing.
        Nodes: 0=(0,0), 1=(1,1), 2=(0,1), 3=(1,0)
        Edges: 0-3 (bottom-left to bottom-right, horizontal) and 2-1 (top-left to top-right, horizontal) -> parallel, no crossing
        -> Changed to an explicit diagonal construction since make_x_graph already covers the original case.

        Uses explicit node_list order for coordinates to avoid ambiguity in networkx node insertion order.
        """
        G = nx.Graph()
        G.add_nodes_from([0, 1, 2, 3])   # register nodes first to guarantee order
        G.add_edges_from([(0, 2), (1, 3)])  # diagonals
        nodes = [0, 1, 2, 3]
        # Node positions: 0=(0,0), 1=(1,0), 2=(1,1), 3=(0,1)
        # Edge (0,2): (0,0)->(1,1); Edge (1,3): (1,0)->(0,1)  -> two diagonals intersect
        coords = np.array([[0.0, 0.0], [1.0, 0.0],
                           [1.0, 1.0], [0.0, 1.0]])
        state = GraphLayoutState(G, nodes, coords)
        count, _ = state.compute_crossings()
        assert count == 1
        assert_cache_consistent(state)


# ── Incremental update consistency (core) ─────────────────────────────────────

class TestIncrementalUpdateConsistency:
    """
    Core tests: after every write operation, the cache must fully match the result computed from scratch.
    """

    def test_update_position_creates_crossing(self):
        """Initially no crossing -> move a node so edges intersect -> detect 1 crossing."""
        G, nodes, coords = make_square_graph()
        state = GraphLayoutState(G, nodes, coords)
        assert state.compute_crossings()[0] == 0

        # Move node 2 (1,1) to (0.1, 0.1), causing edges 1-2 and 3-0 to cross
        state.update_position(2, [0.1, 0.1])
        assert_cache_consistent(state)

    def test_update_position_removes_crossing(self):
        """X-shape -> move a node to eliminate the crossing."""
        G, nodes, coords = make_x_graph()
        state = GraphLayoutState(G, nodes, coords)
        assert state.compute_crossings()[0] == 1

        # Move node 1 (1,1) to (2,2) so the two edges no longer intersect
        state.update_position(1, [2.0, 2.0])
        assert_cache_consistent(state)

    def test_batch_update_consistency(self):
        """Cache and reference must match after batch_update modifies multiple nodes."""
        G, nodes, coords = make_k4_graph()
        state = GraphLayoutState(G, nodes, coords)

        new_coords = np.array([
            [0.0, 0.0], [3.0, 0.0], [1.5, 3.0], [0.5, 1.0]
        ])
        state.batch_update(nodes, new_coords)
        assert_cache_consistent(state)

    def test_apply_deltas_consistency(self):
        """Cache and reference must match after apply_deltas."""
        G, nodes, coords = make_k4_graph()
        state = GraphLayoutState(G, nodes, coords)

        # Apply a small displacement to all nodes
        deltas = np.random.default_rng(42).uniform(-0.1, 0.1, (4, 2))
        state.apply_deltas(nodes, deltas)
        assert_cache_consistent(state)

    def test_multiple_updates_accumulate_correctly(self):
        """Results remain correct after multiple consecutive incremental updates."""
        G, nodes, coords = make_k4_graph()
        state = GraphLayoutState(G, nodes, coords)
        rng = np.random.default_rng(0)

        for _ in range(10):
            idx = rng.integers(0, 4)
            delta = rng.uniform(-0.05, 0.05, 2)
            state.update_position(nodes[idx], state.positions[idx] + delta)

        assert_cache_consistent(state)

    def test_apply_deltas_then_batch_update(self):
        """After apply_deltas, a batch_update should overwrite the delta_valid flag."""
        G, nodes, coords = make_k4_graph()
        state = GraphLayoutState(G, nodes, coords)

        state.apply_deltas(nodes[:2], np.array([[0.1, 0.0], [-0.1, 0.0]]))
        # batch_update should clear delta_valid (absolute write)
        state.batch_update(nodes[2:], np.array([[2.5, 2.5], [0.5, 2.5]]))
        assert_cache_consistent(state)

    def test_no_dirty_after_flush(self):
        """After compute_crossings, the dirty set should be cleared and deltas for flushed nodes should be reset."""
        G, nodes, coords = make_x_graph()
        state = GraphLayoutState(G, nodes, coords)
        state.update_position(nodes[0], [0.1, 0.1])
        assert len(state._dirty_nodes) == 1
        state.compute_crossings()
        # dirty set must be cleared
        assert len(state._dirty_nodes) == 0
        # deltas for flushed node must be zeroed so next move doesn't accumulate
        idx = state.n2i[nodes[0]]
        assert not state._delta_valid[idx]
        np.testing.assert_array_equal(state._node_deltas[idx], [0.0, 0.0])

    def test_idempotent_second_call(self):
        """Two consecutive calls to compute_crossings should return the same result."""
        G, nodes, coords = make_k4_graph()
        state = GraphLayoutState(G, nodes, coords)
        c1, m1 = state.compute_crossings()
        c2, m2 = state.compute_crossings()
        assert c1 == c2
        np.testing.assert_array_equal(m1, m2)


# ── Node unmasking (unmask_node) ──────────────────────────────────────────────

class TestUnmaskNode:
    def test_unmask_reveals_node(self):
        G, nodes, coords = make_x_graph()
        state = GraphLayoutState(G, nodes[:2], coords[:2])
        assert state.num_visible == 2

        state.unmask_node(nodes[2], coords[2])
        assert state.num_visible == 3
        np.testing.assert_array_equal(state.get_position(nodes[2]), coords[2])

    def test_unmask_updates_crossing_detection(self):
        """After revealing all nodes, crossing detection should match full initialization."""
        G, nodes, coords = make_x_graph()
        # Start with only 2 nodes (active edges require both endpoints visible)
        state = GraphLayoutState(G, nodes[:1], coords[:1])
        for node, coord in zip(nodes[1:], coords[1:]):
            state.unmask_node(node, coord)

        assert_cache_consistent(state)
        count, _ = state.compute_crossings()
        assert count == 1

    def test_unmask_already_visible_raises(self):
        G, nodes, coords = make_x_graph()
        state = GraphLayoutState(G, nodes, coords)
        with pytest.raises(ValueError, match="already visible"):
            state.unmask_node(nodes[0], coords[0])

    def test_unmask_then_update_consistent(self):
        """After revealing a node and then moving it, the cache should still be consistent."""
        G, nodes, coords = make_square_graph()
        state = GraphLayoutState(G, nodes[:3], coords[:3])
        state.unmask_node(nodes[3], coords[3])
        state.update_position(nodes[0], np.array([0.5, 0.5]))
        assert_cache_consistent(state)


# ── Error handling ────────────────────────────────────────────────────────────

class TestErrorHandling:
    def test_update_invisible_node_raises(self):
        G, nodes, coords = make_x_graph()
        state = GraphLayoutState(G, nodes[:2], coords[:2])
        with pytest.raises(ValueError, match="not visible"):
            state.update_position(nodes[3], [0.5, 0.5])

    def test_batch_update_invisible_node_raises(self):
        G, nodes, coords = make_x_graph()
        state = GraphLayoutState(G, nodes[:2], coords[:2])
        with pytest.raises(ValueError, match="not visible"):
            state.batch_update(nodes, coords)

    def test_apply_deltas_invisible_node_raises(self):
        G, nodes, coords = make_x_graph()
        state = GraphLayoutState(G, nodes[:2], coords[:2])
        with pytest.raises(ValueError, match="not visible"):
            state.apply_deltas(nodes, np.zeros((4, 2)))


# ── node_dist cache correctness ───────────────────────────────────────────────

class TestNodeDistCache:
    def test_node_dist_initialized_correctly(self):
        """After initialization, _node_dist should equal the manually computed distance matrix."""
        G, nodes, coords = make_x_graph()
        state = GraphLayoutState(G, nodes, coords)
        vis_idx = np.where(state.visible)[0]
        for i in vis_idx:
            for j in vis_idx:
                expected = np.linalg.norm(coords[i] - coords[j])
                assert abs(state._node_dist[i, j] - expected) < 1e-10

    def test_node_dist_updated_after_move(self):
        """After moving a node and flushing, _node_dist should reflect the new coordinates."""
        G, nodes, coords = make_x_graph()
        state = GraphLayoutState(G, nodes, coords)
        new_pos = np.array([0.5, 0.5])
        state.update_position(nodes[0], new_pos)
        state.compute_crossings()  # flush

        for j_node in nodes[1:]:
            j = state.n2i[j_node]
            expected = np.linalg.norm(new_pos - state.positions[j])
            assert abs(state._node_dist[state.n2i[nodes[0]], j] - expected) < 1e-9


# ── Large-graph stress tests ──────────────────────────────────────────────────

class TestStress:
    def test_random_graph_cache_consistency(self):
        """
        Perform several random updates on a random Erdos-Renyi graph and verify cache consistency after each.
        """
        rng = np.random.default_rng(123)
        G = nx.erdos_renyi_graph(20, 0.25, seed=123)
        nodes = list(G.nodes())
        coords = rng.uniform(0, 1, (len(nodes), 2))
        state = GraphLayoutState(G, nodes, coords)

        for _ in range(20):
            # Randomly select 1~3 nodes for batch_update
            k = rng.integers(1, 4)
            chosen = rng.choice(nodes, k, replace=False).tolist()
            new_coords = rng.uniform(0, 1, (k, 2))
            state.batch_update(chosen, new_coords)
            assert_cache_consistent(state)

    def test_incremental_matches_full_rebuild_after_many_moves(self):
        """
        For the same graph, compare incremental updates against full rebuilds after each move;
        the compute_crossings() results must be identical throughout.
        """
        rng = np.random.default_rng(7)
        G = nx.erdos_renyi_graph(12, 0.3, seed=7)
        nodes = list(G.nodes())
        coords = rng.uniform(0, 1, (len(nodes), 2))

        state_inc = GraphLayoutState(G, nodes, coords.copy())

        moves = [
            (rng.choice(nodes), rng.uniform(0, 1, 2))
            for _ in range(15)
        ]

        for node, new_coord in moves:
            state_inc.update_position(node, new_coord)

            # Reference rebuild: compute from scratch using current state_inc coordinates
            ref_state = GraphLayoutState(G, nodes, state_inc.positions.copy())
            inc_count, _ = state_inc.compute_crossings()
            ref_count, _ = ref_state.compute_crossings()
            assert inc_count == ref_count, (
                f"After moving {node} to {new_coord}: inc={inc_count}, ref={ref_count}"
            )


# ── Degenerate geometry: collapse to a line / point ───────────────────────────

class TestDegenerateCollapse:
    """
    Verify that crossing detection is not fooled when nodes are explicitly moved
    to degenerate positions (collinear overlap, all collapsed to one point):
      - Overlapping segments -> collinear-overlap crossings must be detected (Phase 2 collinear_overlap)
      - All nodes at one point -> ALL C(E,2) edge pairs count as crossing (Phase 0)
      - Crossing count after degeneration must be much larger than the original

    On "should approach the product of all node degrees"
    ---------------------------------------------------
    When all N nodes overlap at the same point, for each overlapping node pair (u, v),
    Phase 0 adds deg(u)*deg(v) edge pairs to crossing_mask (after deduplication).
    The exact result is C(E, 2) = E*(E-1)/2, i.e., all E active edges cross each other pairwise.
    This is a tighter exact value than the "degree product" bound and grows rapidly with E.
    """

    # ── Helper: build deterministic star / path / cross graph ────────────────

    @staticmethod
    def _make_cross_graph():
        """
        Cross graph: 5 nodes, center + four arms; 4 edges, all adjacent.
        Node 0=center, 1/2/3/4=arm endpoints.
        Initial layout: center at (0.5,0.5), one arm at each corner.
        """
        G = nx.Graph()
        G.add_nodes_from([0, 1, 2, 3, 4])
        G.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 4)])
        coords = np.array([
            [0.5, 0.5],   # 0 center
            [0.0, 0.5],   # 1 left
            [1.0, 0.5],   # 2 right
            [0.5, 0.0],   # 3 bottom
            [0.5, 1.0],   # 4 top
        ])
        return G, [0, 1, 2, 3, 4], coords

    @staticmethod
    def _make_chord_graph():
        """
        Chord graph: 6 nodes evenly distributed on a regular hexagon,
        with 3 long diameter-crossing chords (0-3, 1-4, 2-5).
        The three chords cross each other pairwise (near the center); initial crossing count = 3.
        """
        G = nx.Graph()
        G.add_nodes_from(range(6))
        G.add_edges_from([(0, 3), (1, 4), (2, 5)])
        angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        coords = np.column_stack([np.cos(angles), np.sin(angles)])
        return G, list(range(6)), coords

    # ── 1. Collinear overlap (segments on same line with overlapping intervals) ──

    def test_collinear_overlap_two_edges(self):
        """
        Two non-adjacent edges both moved onto the x-axis with overlapping intervals
        -> collinear_overlap path detects 1 crossing.

        Layout: nodes 0=(0,0), 1=(1,0), 2=(3,0), 3=(4,0)
        Edge (0,2): x in [0,3]; Edge (1,3): x in [1,4]; overlap interval [1,3] is non-empty -> crossing.
        """
        G = nx.Graph()
        G.add_nodes_from([0, 1, 2, 3])
        G.add_edges_from([(0, 2), (1, 3)])
        nodes = [0, 1, 2, 3]
        # Initial: vertically separated, no crossings
        state = GraphLayoutState(G, nodes,
                                  np.array([[0.0, 0.0], [1.0, 0.0],
                                            [0.0, 1.0], [1.0, 1.0]]))
        assert state.compute_crossings()[0] == 0

        # Collapse to x-axis with overlapping intervals
        state.batch_update(nodes,
                           np.array([[0.0, 0.0], [1.0, 0.0],
                                     [3.0, 0.0], [4.0, 0.0]]))
        count, _ = state.compute_crossings()
        assert count == 1, f"Collinear overlap should detect 1 crossing, got {count}"
        assert_cache_consistent(state)

    def test_collinear_nested_edge_detected(self):
        """
        One long edge completely contains another short edge (nested) -> still collinear_overlap, should be detected.

        Edge (0,3): x in [0,3]; Edge (1,2): x in [1,2], completely inside the former.
        """
        G = nx.Graph()
        G.add_nodes_from([0, 1, 2, 3])
        G.add_edges_from([(0, 3), (1, 2)])
        nodes = [0, 1, 2, 3]
        coords = np.array([[0.0, 0.0], [1.0, 0.0],
                           [2.0, 0.0], [3.0, 0.0]])
        state = GraphLayoutState(G, nodes, coords)
        count, _ = state.compute_crossings()
        assert count == 1, f"Nested collinear edges should detect 1 crossing, got {count}"
        assert_cache_consistent(state)

    def test_collinear_endpoint_touch_not_counted(self):
        """
        Collinear adjacent edges that only touch at endpoints (path graph) -> not counted as crossing.

        Path 0-1-2-3 all moved to the x-axis; three edges meet end-to-end with no interval overlap.
        """
        G = nx.path_graph(4)   # edges: (0,1),(1,2),(2,3)
        nodes = list(G.nodes())
        coords = np.array([[0.0, 0.0], [1.0, 0.0],
                           [2.0, 0.0], [3.0, 0.0]])
        state = GraphLayoutState(G, nodes, coords)
        count, _ = state.compute_crossings()
        assert count == 0, f"Adjacent edges touching only at endpoints should not be counted as crossing, got {count}"

    def test_cross_graph_collapse_arms_to_one_line(self):
        """
        Cross graph: all 4 arm nodes moved to the line through the center node.
        All edges start from the center and are adjacent (share the center), so no extra crossings should be created,
        but when two arm nodes overlap with the center node, Phase 0 detects crossings.
        """
        G, nodes, coords = self._make_cross_graph()
        state = GraphLayoutState(G, nodes, coords)

        # Move all 4 arm endpoints to overlap with the center (0.5, 0.5)
        state.batch_update(
            [1, 2, 3, 4],
            np.full((4, 2), [0.5, 0.5]),
        )
        count, _ = state.compute_crossings()
        # All 5 nodes overlap -> E=4 edges -> C(4,2)=6 crossings
        assert count == 6, f"When all overlap, C(4,2)=6, got {count}"
        assert_cache_consistent(state)

    # ── 2. All nodes collapsed to one point ───────────────────────────────────

    def test_all_nodes_to_single_point_exact_count(self):
        """
        Move all nodes to the same coordinate:
          - All E active edges form Phase-0 overlap crossings pairwise
          - Exact count = C(E,2) = E*(E-1)//2
          - Must be strictly greater than the original crossing count

        Using the square graph (4 nodes, 4 edges, initially no crossings):
          orig=0, collapsed=C(4,2)=6.
        """
        G, nodes, coords = make_square_graph()

        # Originally no crossings
        state_orig = GraphLayoutState(G, nodes, coords.copy())
        orig_count, _ = state_orig.compute_crossings()
        assert orig_count == 0

        # Move all to one point
        state = GraphLayoutState(G, nodes, coords.copy())
        state.batch_update(nodes, np.full((len(nodes), 2), 0.5))

        count, _ = state.compute_crossings()
        E = state.E
        expected = E * (E - 1) // 2  # C(4,2) = 6

        assert count == expected, (
            f"When all nodes overlap, expected C(E,2)=C({E},2)={expected} crossings, got {count}"
        )
        assert count > orig_count, (
            f"After overlap, {count} must be strictly greater than original crossing count {orig_count}"
        )
        assert_cache_consistent(state)

    def test_all_nodes_to_single_point_k4(self):
        """
        K4 (4 nodes, 6 edges) all moved to one point: expected = C(6,2) = 15.
        Also verifies 15 >> original crossing count 1 (square layout has only 1 diagonal crossing).
        """
        G, nodes, coords = make_k4_graph()
        state = GraphLayoutState(G, nodes, coords.copy())
        orig_count, _ = state.compute_crossings()

        state.batch_update(nodes, np.full((len(nodes), 2), 0.5))
        count, _ = state.compute_crossings()

        E = state.E
        expected = E * (E - 1) // 2
        assert count == expected, (
            f"K4 all overlapping should have C(6,2)=15 crossings, got {count}"
        )
        assert count > orig_count * 10, (
            f"After overlap, crossing count {count} should be much larger than original {orig_count} (10x or more)"
        )
        assert_cache_consistent(state)

    def test_single_point_collapse_then_restore(self):
        """
        Move all nodes to one point, then restore to original coordinates: crossing count should recover to original.
        Verify that degenerate state does not contaminate subsequent cache.
        Using square graph (initially 0 crossings, collapsed to C(4,2)=6).
        """
        G, nodes, coords = make_square_graph()
        state = GraphLayoutState(G, nodes, coords.copy())
        orig_count, _ = state.compute_crossings()
        assert orig_count == 0

        # Move all to one point
        state.batch_update(nodes, np.full((len(nodes), 2), 0.5))
        collapsed_count, _ = state.compute_crossings()
        assert collapsed_count > orig_count

        # Restore original coordinates
        state.batch_update(nodes, coords)
        restored_count, _ = state.compute_crossings()
        assert restored_count == orig_count, (
            f"After restoring coordinates, crossing count should be {orig_count}, got {restored_count}"
        )
        assert_cache_consistent(state)

    # ── 3. Partial node overlap ───────────────────────────────────────────────

    def test_two_nodes_overlap_phase0_exact(self):
        """
        Two non-adjacent nodes overlap -> Phase 0 exactly detects deg(u)*deg(v) crossings.

        Graph: 0-1, 0-2, 3-4, 3-5 (two Y-shapes, no common nodes)
        Move node 0 and node 3 to the same position:
          deg(0)=2, deg(3)=2 -> 2*2=4 crossing pairs
        """
        G = nx.Graph()
        G.add_nodes_from(range(6))
        G.add_edges_from([(0, 1), (0, 2), (3, 4), (3, 5)])
        nodes = list(range(6))
        # Initial: two Y-shapes completely separated, no crossings
        coords = np.array([
            [0.0, 0.5], [0.0, 0.0], [0.0, 1.0],   # Y on left
            [2.0, 0.5], [2.0, 0.0], [2.0, 1.0],   # Y on right
        ])
        state = GraphLayoutState(G, nodes, coords)
        assert state.compute_crossings()[0] == 0

        # Node 0 and node 3 overlap
        state.batch_update([0, 3], np.array([[1.0, 0.5], [1.0, 0.5]]))
        count, _ = state.compute_crossings()

        # deg(0)=2, deg(3)=2 -> 4 pairs, no additional geometric crossings
        assert count == 4, (
            f"Two nodes overlap deg=2*2 should detect 4 crossings, got {count}"
        )
        assert_cache_consistent(state)

    def test_incremental_collapse_step_by_step(self):
        """
        Incrementally move chord graph nodes to one point, verify at each step:
          - Crossing count is non-decreasing (more overlap only increases or maintains count)
          - Cache and reference always consistent
        """
        G, nodes, coords = self._make_chord_graph()
        state = GraphLayoutState(G, nodes, coords.copy())
        prev_count, _ = state.compute_crossings()

        target = np.array([0.5, 0.5])
        for node in nodes:
            state.update_position(node, target)
            count, _ = state.compute_crossings()
            assert count >= prev_count, (
                f"After moving node {node} to center, crossing count dropped from {prev_count} to {count}, "
                f"violates monotonicity (overlap should only increase crossings)"
            )
            assert_cache_consistent(state)
            prev_count = count

        # Final result should be C(E,2)
        E = state.E
        assert prev_count == E * (E - 1) // 2


# ── Performance stress test (FPS regression) ─────────────────────────────────
#
# Measures throughput (FPS) for "RL step = update_position + compute_crossings".
# Each scenario writes the measured FPS to tests/fps_baseline.json (creates the file if it doesn't exist).
# When a baseline exists, requires measured FPS to be at least 50% of the baseline,
# to detect severe performance regressions.
#
# Threshold notes
# ---------------
#   REGRESSION_TOLERANCE = 0.50  -> allows at most 50% performance degradation
#   WARMUP_ITERS         = 20    -> warmup iterations, not included in timing
#   BENCH_ITERS          = 200   -> timed iterations (more = more stable)
#
# To update baseline: delete tests/fps_baseline.json and rerun the tests.

_FPS_BASELINE_FILE = os.path.join(os.path.dirname(__file__), "fps_baseline.json")
_REGRESSION_TOLERANCE = 0.50   # allow degradation to 50% of baseline
_WARMUP_ITERS = 20
_BENCH_ITERS  = 200


def _load_fps_baseline() -> dict:
    if os.path.exists(_FPS_BASELINE_FILE):
        with open(_FPS_BASELINE_FILE) as f:
            content = f.read().strip()
            if content:
                return json.loads(content)
    return {}


def _save_fps_baseline(data: dict) -> None:
    with open(_FPS_BASELINE_FILE, "w") as f:
        json.dump(data, f, indent=2)


def _measure_fps(state: GraphLayoutState, nodes: list, rng, iters: int) -> float:
    """
    Execute iters iterations of "randomly move one node + compute_crossings", return FPS.
    """
    n = len(nodes)
    t0 = time.perf_counter()
    for _ in range(iters):
        idx = int(rng.integers(0, n))
        new_pos = rng.uniform(0.0, 1.0, 2)
        state.update_position(nodes[idx], new_pos)
        state.compute_crossings()
    elapsed = time.perf_counter() - t0
    return iters / elapsed


def _make_bench_graph(n_nodes: int, edge_prob: float, seed: int):
    """Build a random graph for stress testing and initialize GraphLayoutState."""
    rng = np.random.default_rng(seed)
    G = nx.erdos_renyi_graph(n_nodes, edge_prob, seed=seed)
    # Ensure at least one connected component; does not affect test purpose
    nodes = list(G.nodes())
    coords = rng.uniform(0.0, 1.0, (len(nodes), 2))
    state = GraphLayoutState(G, nodes, coords)
    return state, nodes


class TestPerformanceFPS:
    """
    FPS stress test:
      - scenario_small  : N=20,  E~40   (small graph, simulates simple env)
      - scenario_medium : N=50,  E~150  (medium graph, typical RL training scenario)
      - scenario_large  : N=100, E~400  (large graph, stress upper bound)

    Each scenario is measured and verified that FPS does not degrade severely (vs. stored baseline).
    """

    @pytest.fixture(autouse=True)
    def _baseline(self):
        """Persist baseline after all tests complete."""
        self._fps_data = _load_fps_baseline()
        yield
        _save_fps_baseline(self._fps_data)

    def _run_scenario(self, key: str, n_nodes: int, edge_prob: float, seed: int):
        state, nodes = _make_bench_graph(n_nodes, edge_prob, seed)
        rng = np.random.default_rng(seed + 1)

        # Warmup
        _measure_fps(state, nodes, rng, _WARMUP_ITERS)

        # Reset rng for reproducibility
        rng = np.random.default_rng(seed + 2)
        fps = _measure_fps(state, nodes, rng, _BENCH_ITERS)

        print(f"\n[FPS] {key}: N={n_nodes}, E={state.E}, FPS={fps:.1f}")

        # Regression check
        if key in self._fps_data:
            baseline_fps = self._fps_data[key]
            threshold = baseline_fps * _REGRESSION_TOLERANCE
            assert fps >= threshold, (
                f"[FPS degradation] {key}: current {fps:.1f} fps < "
                f"baseline {baseline_fps:.1f} x {_REGRESSION_TOLERANCE} "
                f"= {threshold:.1f} fps"
            )
        else:
            # First run: record baseline
            self._fps_data[key] = round(fps, 1)
            print(f"[FPS] {key}: new baseline recorded = {fps:.1f} fps")

    def test_fps_small_graph(self):
        """N=20 small graph, verify FPS basic performance does not degrade."""
        self._run_scenario("small_N20_p020", n_nodes=20, edge_prob=0.20, seed=10)

    def test_fps_medium_graph(self):
        """N=50 medium graph, typical RL training scenario."""
        self._run_scenario("medium_N50_p015", n_nodes=50, edge_prob=0.15, seed=20)

    def test_fps_large_graph(self):
        """N=100 large graph, stress upper bound."""
        self._run_scenario("large_N100_p010", n_nodes=100, edge_prob=0.10, seed=30)

    def test_fps_dense_medium_graph(self):
        """N=30 dense graph (high edge density), crossing detection cost is higher."""
        self._run_scenario("dense_N30_p040", n_nodes=30, edge_prob=0.40, seed=40)

    def test_fps_batch_update_medium_graph(self):
        """
        Throughput of batch_update (update multiple nodes at once) + compute_crossings.
        Move 5 nodes randomly per step, N=50.
        """
        key = "batch5_medium_N50_p015"
        state, nodes = _make_bench_graph(50, 0.15, seed=50)
        rng = np.random.default_rng(51)
        n = len(nodes)

        # Warmup
        for _ in range(_WARMUP_ITERS):
            chosen = rng.choice(nodes, 5, replace=False).tolist()
            state.batch_update(chosen, rng.uniform(0, 1, (5, 2)))
            state.compute_crossings()

        rng = np.random.default_rng(52)
        t0 = time.perf_counter()
        for _ in range(_BENCH_ITERS):
            chosen = rng.choice(nodes, 5, replace=False).tolist()
            state.batch_update(chosen, rng.uniform(0, 1, (5, 2)))
            state.compute_crossings()
        fps = _BENCH_ITERS / (time.perf_counter() - t0)

        print(f"\n[FPS] {key}: N={n}, E={state.E}, FPS={fps:.1f}")

        if key in self._fps_data:
            baseline_fps = self._fps_data[key]
            threshold = baseline_fps * _REGRESSION_TOLERANCE
            assert fps >= threshold, (
                f"[FPS degradation] {key}: current {fps:.1f} < baseline {baseline_fps:.1f} "
                f"x {_REGRESSION_TOLERANCE} = {threshold:.1f} fps"
            )
        else:
            self._fps_data[key] = round(fps, 1)
            print(f"[FPS] {key}: new baseline recorded = {fps:.1f} fps")
