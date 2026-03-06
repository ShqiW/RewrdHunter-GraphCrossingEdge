"""
Softmax Ranking Loss for Structure Consistency

思路：对于每个节点i，计算两个分布：
- P_graph(j|i)  = softmax(-d_graph(i,j) / τ)   # 目标分布（固定）
- P_layout(j|i) = softmax(-d_layout(i,j) / τ)  # 当前分布

Loss = avg_i[ KL(P_graph || P_layout) ]

优点：
- 无需缩放因子（只关注相对顺序）
- 可微分
- 概率解释清晰
"""
import networkx as nx
import torch
import torch.nn.functional as F


class SoftmaxRankingLoss:
    def __init__(
        self,
        G: nx.Graph = None,
        device=None,
        tau: float = None,
        P_graph: torch.Tensor = None,
    ):
        """
        初始化 Softmax Ranking Loss 计算器

        Args:
            G: NetworkX 图 (如果提供，则从图计算P_graph)
            device: torch device
            tau: 温度参数，如果为None则自动设为 mean(d_graph)
            P_graph: 预计算的目标分布 [n, n] (如果提供，则直接使用)
        """
        if device is None:
            device = torch.device("cpu")
        self.device = device

        # 如果提供了预计算的P_graph，直接使用
        if P_graph is not None:
            self.P_graph = P_graph.to(device)
            self.n = P_graph.shape[0]
            self.tau = tau if tau is not None else 1.0
            self.d_graph = None  # 不需要存储
            return

        # 否则从图计算
        if G is None:
            raise ValueError("Must provide either G (graph) or P_graph (precomputed)")

        self.nodes = list(G.nodes())
        self.n = len(self.nodes)

        # 计算图距离（最短路径）
        d_graph = torch.zeros((self.n, self.n), dtype=torch.float32, device=device)
        for i, u in enumerate(self.nodes):
            sp_lengths = nx.single_source_shortest_path_length(G, u)
            for v, dist in sp_lengths.items():
                j = self.nodes.index(v)
                d_graph[i, j] = float(dist)
        self.d_graph = d_graph

        # 设置温度参数
        # 排除对角线（自己到自己的距离=0）
        mask = ~torch.eye(self.n, dtype=torch.bool, device=device)
        if tau is None:
            # 自适应：使用图距离的平均值
            self.tau = d_graph[mask].mean().item()
            if self.tau < 1e-6:
                self.tau = 1.0  # 防止除零
        else:
            self.tau = tau

        # 预计算 P_graph 分布
        # 对每个节点i，计算 P_graph(j|i) = softmax(-d_graph(i,j) / τ)
        # 需要排除自己 (j != i)
        self.P_graph = self._compute_softmax_distribution(d_graph)

    def _compute_softmax_distribution(self, distances: torch.Tensor) -> torch.Tensor:
        """
        计算每个节点的softmax分布

        Args:
            distances: [n, n] 距离矩阵

        Returns:
            P: [n, n] 概率分布，P[i,j] = P(j|i)，对角线为0
        """
        n = distances.shape[0]

        # logits = -d / τ
        logits = -distances / self.tau

        # 对角线设为 -inf（排除自己）
        mask = torch.eye(n, dtype=torch.bool, device=self.device)
        logits = logits.masked_fill(mask, float('-inf'))

        # softmax
        P = F.softmax(logits, dim=1)

        return P

    def _compute_layout_distances(self, coords: torch.Tensor) -> torch.Tensor:
        """
        计算布局中的欧氏距离矩阵

        Args:
            coords: [n, 2] 坐标

        Returns:
            d_layout: [n, n] 距离矩阵
        """
        # d_layout[i,j] = ||coords[i] - coords[j]||
        squared_norms = (coords ** 2).sum(dim=1)
        d_sq = squared_norms[:, None] + squared_norms[None, :] - 2 * coords @ coords.T
        d_layout = torch.sqrt(torch.clamp(d_sq, min=1e-8))
        return d_layout

    def compute_kl(self, coords: torch.Tensor) -> torch.Tensor:
        """
        计算 KL(P_graph || P_layout)

        Args:
            coords: [n, 2] 当前布局坐标

        Returns:
            kl: 标量，平均KL散度
        """
        # 计算布局距离
        d_layout = self._compute_layout_distances(coords)

        # 计算 P_layout
        P_layout = self._compute_softmax_distribution(d_layout)

        # KL(P_graph || P_layout) = Σ P_graph * log(P_graph / P_layout)
        # = Σ P_graph * (log P_graph - log P_layout)

        # 避免 log(0)
        eps = 1e-8
        log_P_graph = torch.log(self.P_graph + eps)
        log_P_layout = torch.log(P_layout + eps)

        # 对角线是0，不参与计算
        mask = ~torch.eye(self.n, dtype=torch.bool, device=self.device)

        # KL per node
        kl_per_node = (self.P_graph * (log_P_graph - log_P_layout)).sum(dim=1)

        # 平均
        kl = kl_per_node.mean()

        return kl

    def __call__(self, coords: torch.Tensor) -> torch.Tensor:
        """
        计算 Softmax Ranking Loss

        Args:
            coords: [n, 2] 坐标

        Returns:
            loss: KL散度（越小越好）
        """
        return self.compute_kl(coords)

    def get_tau(self) -> float:
        """返回温度参数"""
        return self.tau
