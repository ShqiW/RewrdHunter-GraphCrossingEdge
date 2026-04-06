"""
Tests for GraphLayoutState — cache correctness and crossing detection.

Test strategy
-------------
每个测试用例都有一个几何上明确的图（知道有几条交叉边），
并通过以下两种方式验证 cache 的正确性：
  1. 直接调用 compute_crossings() 的结果与预期值对比
  2. 增量更新后的结果与"全量重建"后的结果对比（一致性检验）
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
    X 形图: 两条对角线交叉
    节点: 0=(0,0), 1=(1,1), 2=(0,1), 3=(1,0)
    边  : 0-1, 2-3  → 在 (0.5,0.5) 处恰好相交，crossing_count = 1
    """
    G = nx.Graph()
    G.add_edges_from([(0, 1), (2, 3)])
    coords = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    return G, list(G.nodes()), coords


def make_square_graph():
    """
    正方形图: 四条边，无交叉
    节点: 0=(0,0), 1=(1,0), 2=(1,1), 3=(0,1)
    边  : 0-1, 1-2, 2-3, 3-0
    """
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    return G, list(G.nodes()), coords


def make_k4_graph():
    """
    K4 完全图（4节点）用平面嵌入 → 0 交叉
    节点: 0=(0,0), 1=(2,0), 2=(1,2), 3=(1,0.5) (中心点)
    平面图，K4 可嵌入平面，这里用已知无交叉坐标。
    """
    G = nx.complete_graph(4)
    # 在正方形内嵌入，不产生交叉的已知坐标
    coords = np.array([
        [0.0, 0.0],  # 0
        [2.0, 0.0],  # 1
        [2.0, 2.0],  # 2
        [0.0, 2.0],  # 3
    ])
    # K4 在这个布局里会有交叉 (对角线 0-2 和 1-3 相交)，这正是我们要测的
    return G, list(G.nodes()), coords


def reference_crossing_count(state: GraphLayoutState) -> int:
    """用 precompute_geometry 从零计算交叉数（不走 cache）。"""
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
    验证增量 cache 与从零计算的结果完全一致。
    同时检验 _node_dist、_half_lens、_midpoints、_mid_dist 的内部一致性。
    """
    # 先 flush（触发增量更新）
    count_inc, mask_inc = state.compute_crossings()
    count_ref = reference_crossing_count(state)
    assert count_inc == count_ref, (
        f"交叉数不一致: cache={count_inc}, reference={count_ref}"
    )

    # 验证 _node_dist 对可见节点的对称性
    vis = np.where(state.visible)[0]
    d = state._node_dist[np.ix_(vis, vis)]
    assert np.allclose(d, d.T, atol=1e-10), "_node_dist 不对称"
    assert np.allclose(np.diag(d), 0.0, atol=1e-10), "_node_dist 对角线非零"

    # 验证 _half_lens 与实际坐标一致
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
            f"edge {e_idx}: midpoint cache错误"
        )


# ── 初始化测试 ─────────────────────────────────────────────────────────────────

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
        """只有部分节点可见时，初始化应正常工作。"""
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


# ── 交叉检测正确性 ─────────────────────────────────────────────────────────────

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
        """K4 的正方形四角布局有 1 条交叉（对角线 0-2 与 1-3）。"""
        G, nodes, coords = make_k4_graph()
        state = GraphLayoutState(G, nodes, coords)
        count, _ = state.compute_crossings()
        assert_cache_consistent(state)
        # 正方形 K4: 只有对角线 {0,2} 和 {1,3} 相交
        assert count == 1

    def test_crossing_mask_upper_triangular(self):
        G, nodes, coords = make_x_graph()
        state = GraphLayoutState(G, nodes, coords)
        _, mask = state.compute_crossings()
        # 下三角必须全为 False
        lower = np.tril(mask, k=-1)
        assert not lower.any()

    def test_adjacent_edges_not_counted(self):
        """共享端点的边不应被计为交叉。"""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2)])
        coords = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 0.0]])
        state = GraphLayoutState(G, list(G.nodes()), coords)
        count, _ = state.compute_crossings()
        assert count == 0

    def test_parallel_non_overlapping_no_crossing(self):
        """平行但不重叠的两条边不应被判为交叉。"""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (2, 3)])
        coords = np.array([[0.0, 0.0], [1.0, 0.0],
                           [0.0, 1.0], [1.0, 1.0]])
        state = GraphLayoutState(G, list(G.nodes()), coords)
        count, _ = state.compute_crossings()
        assert count == 0

    def test_two_diagonal_crossings(self):
        """
        两条对角线（X 形的变体，用 make_x_graph 节点但不同边），确认有 1 条交叉。
        节点: 0=(0,0), 1=(1,1), 2=(0,1), 3=(1,0)
        边  : 0-3 (从左下到右下, 水平) 与 2-1 (从左上到右上，水平) → 平行不交叉
        → 改为明确测试 make_x_graph 已覆盖，这里改为对角线明确构造。

        明确用 node_list 顺序传入坐标，避免 networkx 节点插入顺序的歧义。
        """
        G = nx.Graph()
        G.add_nodes_from([0, 1, 2, 3])   # 先注册节点，保证顺序
        G.add_edges_from([(0, 2), (1, 3)])  # 对角线
        nodes = [0, 1, 2, 3]
        # 节点位置: 0=(0,0), 1=(1,0), 2=(1,1), 3=(0,1)
        # 边 (0,2): (0,0)→(1,1)；边 (1,3): (1,0)→(0,1)  → 两对角线相交
        coords = np.array([[0.0, 0.0], [1.0, 0.0],
                           [1.0, 1.0], [0.0, 1.0]])
        state = GraphLayoutState(G, nodes, coords)
        count, _ = state.compute_crossings()
        assert count == 1
        assert_cache_consistent(state)


# ── 增量更新一致性（核心） ─────────────────────────────────────────────────────

class TestIncrementalUpdateConsistency:
    """
    核心测试：每次写操作后，cache 与从零计算的结果完全一致。
    """

    def test_update_position_creates_crossing(self):
        """初始无交叉 → 移动节点使边相交 → 检测到 1 条交叉。"""
        G, nodes, coords = make_square_graph()
        state = GraphLayoutState(G, nodes, coords)
        assert state.compute_crossings()[0] == 0

        # 把节点 2 (1,1) 移到 (0.1, 0.1)，使边 1-2 和 3-0 交叉
        state.update_position(2, [0.1, 0.1])
        assert_cache_consistent(state)

    def test_update_position_removes_crossing(self):
        """X 形 → 移动节点消除交叉。"""
        G, nodes, coords = make_x_graph()
        state = GraphLayoutState(G, nodes, coords)
        assert state.compute_crossings()[0] == 1

        # 把节点 1 (1,1) 移到 (2,2)，使两条边不再相交
        state.update_position(1, [2.0, 2.0])
        assert_cache_consistent(state)

    def test_batch_update_consistency(self):
        """batch_update 更新多个节点后 cache 与 reference 一致。"""
        G, nodes, coords = make_k4_graph()
        state = GraphLayoutState(G, nodes, coords)

        new_coords = np.array([
            [0.0, 0.0], [3.0, 0.0], [1.5, 3.0], [0.5, 1.0]
        ])
        state.batch_update(nodes, new_coords)
        assert_cache_consistent(state)

    def test_apply_deltas_consistency(self):
        """apply_deltas 后 cache 与 reference 一致。"""
        G, nodes, coords = make_k4_graph()
        state = GraphLayoutState(G, nodes, coords)

        # 对所有节点施加一个小位移
        deltas = np.random.default_rng(42).uniform(-0.1, 0.1, (4, 2))
        state.apply_deltas(nodes, deltas)
        assert_cache_consistent(state)

    def test_multiple_updates_accumulate_correctly(self):
        """连续多次增量更新后结果仍正确。"""
        G, nodes, coords = make_k4_graph()
        state = GraphLayoutState(G, nodes, coords)
        rng = np.random.default_rng(0)

        for _ in range(10):
            idx = rng.integers(0, 4)
            delta = rng.uniform(-0.05, 0.05, 2)
            state.update_position(nodes[idx], state.positions[idx] + delta)

        assert_cache_consistent(state)

    def test_apply_deltas_then_batch_update(self):
        """apply_deltas 之后再 batch_update，覆盖 delta_valid 标志。"""
        G, nodes, coords = make_k4_graph()
        state = GraphLayoutState(G, nodes, coords)

        state.apply_deltas(nodes[:2], np.array([[0.1, 0.0], [-0.1, 0.0]]))
        # batch_update 应清空 delta_valid（绝对写）
        state.batch_update(nodes[2:], np.array([[2.5, 2.5], [0.5, 2.5]]))
        assert_cache_consistent(state)

    def test_no_dirty_after_flush(self):
        """compute_crossings 之后 dirty set 清空，已 flush 节点的 delta 也应重置。"""
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
        """连续两次 compute_crossings 结果相同。"""
        G, nodes, coords = make_k4_graph()
        state = GraphLayoutState(G, nodes, coords)
        c1, m1 = state.compute_crossings()
        c2, m2 = state.compute_crossings()
        assert c1 == c2
        np.testing.assert_array_equal(m1, m2)


# ── 节点揭示 (unmask_node) ────────────────────────────────────────────────────

class TestUnmaskNode:
    def test_unmask_reveals_node(self):
        G, nodes, coords = make_x_graph()
        state = GraphLayoutState(G, nodes[:2], coords[:2])
        assert state.num_visible == 2

        state.unmask_node(nodes[2], coords[2])
        assert state.num_visible == 3
        np.testing.assert_array_equal(state.get_position(nodes[2]), coords[2])

    def test_unmask_updates_crossing_detection(self):
        """揭示所有节点后，交叉检测结果应与全量初始化一致。"""
        G, nodes, coords = make_x_graph()
        # 先只放 2 个节点（不构成活跃边的两端点都可见）
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
        """先揭示节点，再移动，cache 仍一致。"""
        G, nodes, coords = make_square_graph()
        state = GraphLayoutState(G, nodes[:3], coords[:3])
        state.unmask_node(nodes[3], coords[3])
        state.update_position(nodes[0], np.array([0.5, 0.5]))
        assert_cache_consistent(state)


# ── 错误处理 ──────────────────────────────────────────────────────────────────

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


# ── node_dist cache 正确性 ────────────────────────────────────────────────────

class TestNodeDistCache:
    def test_node_dist_initialized_correctly(self):
        """初始化后 _node_dist 应等于手工计算的距离矩阵。"""
        G, nodes, coords = make_x_graph()
        state = GraphLayoutState(G, nodes, coords)
        vis_idx = np.where(state.visible)[0]
        for i in vis_idx:
            for j in vis_idx:
                expected = np.linalg.norm(coords[i] - coords[j])
                assert abs(state._node_dist[i, j] - expected) < 1e-10

    def test_node_dist_updated_after_move(self):
        """移动节点后 flush，_node_dist 应反映新坐标。"""
        G, nodes, coords = make_x_graph()
        state = GraphLayoutState(G, nodes, coords)
        new_pos = np.array([0.5, 0.5])
        state.update_position(nodes[0], new_pos)
        state.compute_crossings()  # flush

        for j_node in nodes[1:]:
            j = state.n2i[j_node]
            expected = np.linalg.norm(new_pos - state.positions[j])
            assert abs(state._node_dist[state.n2i[nodes[0]], j] - expected) < 1e-9


# ── 大图压力测试 ──────────────────────────────────────────────────────────────

class TestStress:
    def test_random_graph_cache_consistency(self):
        """
        对随机 Erdős–Rényi 图做若干次随机更新，每次都验证 cache 一致性。
        """
        rng = np.random.default_rng(123)
        G = nx.erdos_renyi_graph(20, 0.25, seed=123)
        nodes = list(G.nodes())
        coords = rng.uniform(0, 1, (len(nodes), 2))
        state = GraphLayoutState(G, nodes, coords)

        for _ in range(20):
            # 随机选 1~3 个节点做 batch_update
            k = rng.integers(1, 4)
            chosen = rng.choice(nodes, k, replace=False).tolist()
            new_coords = rng.uniform(0, 1, (k, 2))
            state.batch_update(chosen, new_coords)
            assert_cache_consistent(state)

    def test_incremental_matches_full_rebuild_after_many_moves(self):
        """
        对同一个图，一份用增量更新，一份每次全量重建，
        经过多轮移动后两者的 compute_crossings() 结果完全一致。
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

            # 重建参照：使用当前 state_inc 的坐标从头算
            ref_state = GraphLayoutState(G, nodes, state_inc.positions.copy())
            inc_count, _ = state_inc.compute_crossings()
            ref_count, _ = ref_state.compute_crossings()
            assert inc_count == ref_count, (
                f"移动 {node} 到 {new_coord} 后: inc={inc_count}, ref={ref_count}"
            )


# ── 退化几何：塌缩到线 / 点 ────────────────────────────────────────────────────

class TestDegenerateCollapse:
    """
    验证当节点被显式移动到退化位置（共线重叠、全部重叠于一点）时，
    交叉检测不会被愚弄：
      - 重叠的线段 → 共线重叠交叉必须被检出（Phase 2 collinear_overlap）
      - 所有节点重叠于一点 → ALL C(E,2) 条边对均计为交叉（Phase 0）
      - 退化后的交叉数必须远大于原图

    关于"应接近所有点的 degree 的叠乘"
    ------------------------------------
    当所有 N 个节点重叠于同一点时，对每个相互重叠的节点对 (u, v)，
    Phase 0 把 deg(u)×deg(v) 条边对加入 crossing_mask（mask 去重）。
    最终精确结果为 C(E, 2) = E*(E-1)/2，即所有 E 条活跃边两两相交。
    这是一个比"degree 叠乘"更紧的精确值，且随 E 快速增长。
    """

    # ── 辅助：构建确定性 star / path / cross 图 ──────────────────────────────

    @staticmethod
    def _make_cross_graph():
        """
        十字图：5 个节点，中心 + 四臂；4 条边，全部相邻。
        节点 0=中心, 1/2/3/4=臂端。
        初始布局: 中心(0.5,0.5), 四角各一臂。
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
        弦图：6 个节点均匀排布在正六边形上，
        加上 3 条穿越直径的长弦 (0-3, 1-4, 2-5)。
        三条弦两两相交（在中心附近），初始交叉数 = 3。
        """
        G = nx.Graph()
        G.add_nodes_from(range(6))
        G.add_edges_from([(0, 3), (1, 4), (2, 5)])
        angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        coords = np.column_stack([np.cos(angles), np.sin(angles)])
        return G, list(range(6)), coords

    # ── 1. 共线重叠（线段在同一直线上且区间重叠）────────────────────────────

    def test_collinear_overlap_two_edges(self):
        """
        两条不相邻边都移到 x 轴上，且区间重叠 → collinear_overlap 路径检出 1 条交叉。

        布局: 节点 0=(0,0), 1=(1,0), 2=(3,0), 3=(4,0)
        边 (0,2): x∈[0,3]; 边 (1,3): x∈[1,4]; 重叠区间 [1,3] 非空 → 交叉。
        """
        G = nx.Graph()
        G.add_nodes_from([0, 1, 2, 3])
        G.add_edges_from([(0, 2), (1, 3)])
        nodes = [0, 1, 2, 3]
        # 初始: 竖向分开，无交叉
        state = GraphLayoutState(G, nodes,
                                  np.array([[0.0, 0.0], [1.0, 0.0],
                                            [0.0, 1.0], [1.0, 1.0]]))
        assert state.compute_crossings()[0] == 0

        # 塌缩到 x 轴，区间重叠
        state.batch_update(nodes,
                           np.array([[0.0, 0.0], [1.0, 0.0],
                                     [3.0, 0.0], [4.0, 0.0]]))
        count, _ = state.compute_crossings()
        assert count == 1, f"共线重叠应检出 1 条交叉，实际 {count}"
        assert_cache_consistent(state)

    def test_collinear_nested_edge_detected(self):
        """
        一条长边完全包含另一条短边（嵌套）→ 仍属于 collinear_overlap，应检出。

        边 (0,3): x∈[0,3]; 边 (1,2): x∈[1,2]，完全在前者内部。
        """
        G = nx.Graph()
        G.add_nodes_from([0, 1, 2, 3])
        G.add_edges_from([(0, 3), (1, 2)])
        nodes = [0, 1, 2, 3]
        coords = np.array([[0.0, 0.0], [1.0, 0.0],
                           [2.0, 0.0], [3.0, 0.0]])
        state = GraphLayoutState(G, nodes, coords)
        count, _ = state.compute_crossings()
        assert count == 1, f"嵌套共线边应检出 1 条交叉，实际 {count}"
        assert_cache_consistent(state)

    def test_collinear_endpoint_touch_not_counted(self):
        """
        共线且只在端点相接的相邻边（路径图）→ 不算交叉。

        路径 0-1-2-3 全部移到 x 轴，三条边首尾相接，无区间重叠。
        """
        G = nx.path_graph(4)   # edges: (0,1),(1,2),(2,3)
        nodes = list(G.nodes())
        coords = np.array([[0.0, 0.0], [1.0, 0.0],
                           [2.0, 0.0], [3.0, 0.0]])
        state = GraphLayoutState(G, nodes, coords)
        count, _ = state.compute_crossings()
        assert count == 0, f"端点相接的相邻边不应计为交叉，实际 {count}"

    def test_cross_graph_collapse_arms_to_one_line(self):
        """
        十字图：4 臂节点全部移到中心节点所在直线上。
        所有边都从中心出发，相邻（共享中心），故不应产生额外交叉，
        但当两臂节点与中心节点三点重叠时，Phase 0 会检出交叉。
        """
        G, nodes, coords = self._make_cross_graph()
        state = GraphLayoutState(G, nodes, coords)

        # 把全部 4 个臂端移到与中心 (0.5, 0.5) 重叠的位置
        state.batch_update(
            [1, 2, 3, 4],
            np.full((4, 2), [0.5, 0.5]),
        )
        count, _ = state.compute_crossings()
        # 5 个节点全部重叠 → E=4 条边 → C(4,2)=6 条交叉
        assert count == 6, f"全部重叠时 C(4,2)=6，实际 {count}"
        assert_cache_consistent(state)

    # ── 2. 所有节点重叠于一点 ────────────────────────────────────────────────

    def test_all_nodes_to_single_point_exact_count(self):
        """
        将图所有节点移到同一坐标点：
          - 所有 E 条活跃边两两形成 Phase-0 重叠交叉
          - 精确值 = C(E,2) = E*(E-1)//2
          - 必须严格大于原始交叉数

        使用正方形图（4节点，4条边，初始无交叉）：
          orig=0, collapsed=C(4,2)=6。
        """
        G, nodes, coords = make_square_graph()

        # 原始无交叉
        state_orig = GraphLayoutState(G, nodes, coords.copy())
        orig_count, _ = state_orig.compute_crossings()
        assert orig_count == 0

        # 全部移到同一点
        state = GraphLayoutState(G, nodes, coords.copy())
        state.batch_update(nodes, np.full((len(nodes), 2), 0.5))

        count, _ = state.compute_crossings()
        E = state.E
        expected = E * (E - 1) // 2  # C(4,2) = 6

        assert count == expected, (
            f"所有节点重叠时应有 C(E,2)=C({E},2)={expected} 条交叉，实际 {count}"
        )
        assert count > orig_count, (
            f"重叠后 {count} 必须严格大于原始交叉数 {orig_count}"
        )
        assert_cache_consistent(state)

    def test_all_nodes_to_single_point_k4(self):
        """
        K4（4 节点 6 边）全部移到一点：expected = C(6,2) = 15。
        同时验证 15 >> 原始交叉数 1（正方形布局只有 1 条对角线交叉）。
        """
        G, nodes, coords = make_k4_graph()
        state = GraphLayoutState(G, nodes, coords.copy())
        orig_count, _ = state.compute_crossings()

        state.batch_update(nodes, np.full((len(nodes), 2), 0.5))
        count, _ = state.compute_crossings()

        E = state.E
        expected = E * (E - 1) // 2
        assert count == expected, (
            f"K4 全部重叠应有 C(6,2)=15 条交叉，实际 {count}"
        )
        assert count > orig_count * 10, (
            f"重叠后交叉数 {count} 应远大于原始 {orig_count}（10× 以上）"
        )
        assert_cache_consistent(state)

    def test_single_point_collapse_then_restore(self):
        """
        将节点全部移到一点，再移回原坐标：交叉数应恢复到原始值。
        验证退化状态不会"污染"后续 cache。
        使用正方形图（初始 0 交叉，塌缩后 C(4,2)=6）。
        """
        G, nodes, coords = make_square_graph()
        state = GraphLayoutState(G, nodes, coords.copy())
        orig_count, _ = state.compute_crossings()
        assert orig_count == 0

        # 全部移到一点
        state.batch_update(nodes, np.full((len(nodes), 2), 0.5))
        collapsed_count, _ = state.compute_crossings()
        assert collapsed_count > orig_count

        # 恢复原坐标
        state.batch_update(nodes, coords)
        restored_count, _ = state.compute_crossings()
        assert restored_count == orig_count, (
            f"恢复原坐标后交叉数应为 {orig_count}，实际 {restored_count}"
        )
        assert_cache_consistent(state)

    # ── 3. 部分节点重叠 ──────────────────────────────────────────────────────

    def test_two_nodes_overlap_phase0_exact(self):
        """
        两个非相邻节点重叠 → Phase 0 精确检出 deg(u)×deg(v) 条交叉。

        图: 0-1, 0-2, 3-4, 3-5（两组 Y 形，无公共节点）
        把节点 0 和节点 3 移到同一位置：
          deg(0)=2, deg(3)=2 → 2×2=4 条交叉对
        """
        G = nx.Graph()
        G.add_nodes_from(range(6))
        G.add_edges_from([(0, 1), (0, 2), (3, 4), (3, 5)])
        nodes = list(range(6))
        # 初始: 两个 Y 形完全分开，无交叉
        coords = np.array([
            [0.0, 0.5], [0.0, 0.0], [0.0, 1.0],   # Y on left
            [2.0, 0.5], [2.0, 0.0], [2.0, 1.0],   # Y on right
        ])
        state = GraphLayoutState(G, nodes, coords)
        assert state.compute_crossings()[0] == 0

        # 节点 0 和节点 3 重叠
        state.batch_update([0, 3], np.array([[1.0, 0.5], [1.0, 0.5]]))
        count, _ = state.compute_crossings()

        # deg(0)=2, deg(3)=2 → 4 对，无额外几何交叉
        assert count == 4, (
            f"两节点重叠 deg=2×2 应检出 4 条交叉，实际 {count}"
        )
        assert_cache_consistent(state)

    def test_incremental_collapse_step_by_step(self):
        """
        逐步把弦图节点移到一点，每步都验证：
          - 交叉数单调不减（更多重叠只会增加或保持交叉数）
          - cache 与 reference 始终一致
        """
        G, nodes, coords = self._make_chord_graph()
        state = GraphLayoutState(G, nodes, coords.copy())
        prev_count, _ = state.compute_crossings()

        target = np.array([0.5, 0.5])
        for node in nodes:
            state.update_position(node, target)
            count, _ = state.compute_crossings()
            assert count >= prev_count, (
                f"移动节点 {node} 到中心后交叉数从 {prev_count} 降至 {count}，"
                f"违反单调性（重叠只应增加交叉）"
            )
            assert_cache_consistent(state)
            prev_count = count

        # 最终应为 C(E,2)
        E = state.E
        assert prev_count == E * (E - 1) // 2


# ── 性能压力测试（FPS 回归） ─────────────────────────────────────────────────────
#
# 测量"RL 步 = update_position + compute_crossings"的吞吐量（FPS）。
# 每个 scenario 会把实测 FPS 写入 tests/fps_baseline.json（如果文件不存在则新建）。
# 当 baseline 已存在时，要求实测 FPS 不低于 baseline 的 50%，以检测剧烈性能退化。
#
# 阈值说明
# --------
#   REGRESSION_TOLERANCE = 0.50  → 允许最多 50% 的性能下降
#   WARMUP_ITERS         = 20    → 预热轮次，不计入计时
#   BENCH_ITERS          = 200   → 计时轮次（越多越稳定）
#
# 如何更新 baseline：删除 tests/fps_baseline.json，重新运行测试即可。

_FPS_BASELINE_FILE = os.path.join(os.path.dirname(__file__), "fps_baseline.json")
_REGRESSION_TOLERANCE = 0.50   # 允许退化至 baseline 的 50%
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
    执行 iters 次"随机移动单节点 + compute_crossings"，返回 FPS。
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
    """构建压力测试用随机图并初始化 GraphLayoutState。"""
    rng = np.random.default_rng(seed)
    G = nx.erdos_renyi_graph(n_nodes, edge_prob, seed=seed)
    # 确保至少有一个连通分量，不影响测试目的
    nodes = list(G.nodes())
    coords = rng.uniform(0.0, 1.0, (len(nodes), 2))
    state = GraphLayoutState(G, nodes, coords)
    return state, nodes


class TestPerformanceFPS:
    """
    FPS 压力测试：
      - scenario_small  : N=20,  E≈~40   （小图，模拟简单环境）
      - scenario_medium : N=50,  E≈~150  （中图，典型 RL 训练场景）
      - scenario_large  : N=100, E≈~400  （大图，压力上限）

    每个 scenario 分别测量并校验 FPS 不剧烈退化（vs. 已存 baseline）。
    """

    @pytest.fixture(autouse=True)
    def _baseline(self):
        """在所有测试结束后统一持久化 baseline。"""
        self._fps_data = _load_fps_baseline()
        yield
        _save_fps_baseline(self._fps_data)

    def _run_scenario(self, key: str, n_nodes: int, edge_prob: float, seed: int):
        state, nodes = _make_bench_graph(n_nodes, edge_prob, seed)
        rng = np.random.default_rng(seed + 1)

        # 预热
        _measure_fps(state, nodes, rng, _WARMUP_ITERS)

        # 重置 rng 保证可复现
        rng = np.random.default_rng(seed + 2)
        fps = _measure_fps(state, nodes, rng, _BENCH_ITERS)

        print(f"\n[FPS] {key}: N={n_nodes}, E={state.E}, FPS={fps:.1f}")

        # 回归检验
        if key in self._fps_data:
            baseline_fps = self._fps_data[key]
            threshold = baseline_fps * _REGRESSION_TOLERANCE
            assert fps >= threshold, (
                f"[FPS 退化] {key}: 当前 {fps:.1f} fps < "
                f"baseline {baseline_fps:.1f} × {_REGRESSION_TOLERANCE} "
                f"= {threshold:.1f} fps"
            )
        else:
            # 首次运行：记录 baseline
            self._fps_data[key] = round(fps, 1)
            print(f"[FPS] {key}: 已记录新 baseline = {fps:.1f} fps")

    def test_fps_small_graph(self):
        """N=20 小图，验证 FPS 基本性能不退化。"""
        self._run_scenario("small_N20_p020", n_nodes=20, edge_prob=0.20, seed=10)

    def test_fps_medium_graph(self):
        """N=50 中图，典型 RL 训练场景。"""
        self._run_scenario("medium_N50_p015", n_nodes=50, edge_prob=0.15, seed=20)

    def test_fps_large_graph(self):
        """N=100 大图，压力上限。"""
        self._run_scenario("large_N100_p010", n_nodes=100, edge_prob=0.10, seed=30)

    def test_fps_dense_medium_graph(self):
        """N=30 密集图（高边密度），交叉检测 cost 更高。"""
        self._run_scenario("dense_N30_p040", n_nodes=30, edge_prob=0.40, seed=40)

    def test_fps_batch_update_medium_graph(self):
        """
        batch_update（一次更新多节点） + compute_crossings 的吞吐。
        每步随机移动 5 个节点，N=50。
        """
        key = "batch5_medium_N50_p015"
        state, nodes = _make_bench_graph(50, 0.15, seed=50)
        rng = np.random.default_rng(51)
        n = len(nodes)

        # 预热
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
                f"[FPS 退化] {key}: 当前 {fps:.1f} < baseline {baseline_fps:.1f} "
                f"× {_REGRESSION_TOLERANCE} = {threshold:.1f} fps"
            )
        else:
            self._fps_data[key] = round(fps, 1)
            print(f"[FPS] {key}: 已记录新 baseline = {fps:.1f} fps")
