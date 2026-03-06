"""
Rank Matching Loss for Structure Consistency.

Intuition: For each node, the ranking of other nodes by distance
should be the same in the graph and the layout.

Uses differentiable soft ranking via softmax.
"""
import networkx as nx
import torch
import torch.nn.functional as F


class RankMatchingLoss:
    """
    Rank Matching loss using soft ranking.

    For each node i:
    - Compute soft ranks based on d_graph (target)
    - Compute soft ranks based on d_layout (current)
    - Measure alignment between rankings

    Advantage: Invariant to global scaling (only relative order matters)
    """

    def __init__(self, G: nx.Graph, device=None, tau: float = None):
        """
        Args:
            G: NetworkX graph
            device: torch device
            tau: Temperature for soft ranking. None = adaptive
        """
        if device is None:
            device = torch.device("cpu")
        self.device = device

        self.nodes = list(G.nodes())
        self.n = len(self.nodes)

        # Compute graph distances
        d_graph = torch.zeros((self.n, self.n), dtype=torch.float32, device=device)
        for i, u in enumerate(self.nodes):
            sp_lengths = nx.single_source_shortest_path_length(G, u)
            for v, dist in sp_lengths.items():
                j = self.nodes.index(v)
                d_graph[i, j] = float(dist)
        self.d_graph = d_graph

        # Compute tau
        mask = ~torch.eye(self.n, dtype=torch.bool, device=device)
        if tau is None:
            self.tau = d_graph[mask].mean().item()
            if self.tau < 1e-6:
                self.tau = 1.0
        else:
            self.tau = tau

        # Precompute soft ranks for graph distances (target)
        # soft_rank[i, j] = "how close is j to i" (higher = closer)
        self.soft_rank_graph = self._compute_soft_ranks(d_graph)

    def _compute_soft_ranks(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Compute soft ranking using softmax over negative distances.

        Args:
            distances: [n, n] distance matrix

        Returns:
            soft_ranks: [n, n] where soft_ranks[i, j] indicates
                        the soft rank of j relative to i (higher = closer)
        """
        n = distances.shape[0]

        # Use negative distance as logits (closer = higher score)
        logits = -distances / self.tau

        # Mask out self (diagonal)
        diag_mask = torch.eye(n, dtype=torch.bool, device=self.device)
        logits = logits.masked_fill(diag_mask, float('-inf'))

        # Softmax gives "probability of being the closest"
        soft_ranks = F.softmax(logits, dim=1)

        return soft_ranks

    def _compute_layout_distances(self, coords: torch.Tensor) -> torch.Tensor:
        """Compute pairwise Euclidean distances."""
        squared_norms = (coords ** 2).sum(dim=1)
        d_sq = squared_norms[:, None] + squared_norms[None, :] - 2 * coords @ coords.T
        d_layout = torch.sqrt(torch.clamp(d_sq, min=1e-8))
        return d_layout

    def __call__(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Compute rank matching loss.

        Uses MSE between soft rankings (could also use KL or cosine).

        Args:
            coords: [n, 2] node coordinates

        Returns:
            loss: scalar, lower is better
        """
        # Compute layout distances and soft ranks
        d_layout = self._compute_layout_distances(coords)
        soft_rank_layout = self._compute_soft_ranks(d_layout)

        # MSE between soft rankings
        loss = F.mse_loss(soft_rank_layout, self.soft_rank_graph)

        return loss

    def get_tau(self) -> float:
        """Return temperature parameter."""
        return self.tau
