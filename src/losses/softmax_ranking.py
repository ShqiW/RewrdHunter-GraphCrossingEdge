"""
Softmax Ranking Loss for Structure Consistency

Idea: For each node i, compute two distributions:
- P_graph(j|i)  = softmax(-d_graph(i,j) / tau)   # target distribution (fixed)
- P_layout(j|i) = softmax(-d_layout(i,j) / tau)  # current distribution

Loss = avg_i[ KL(P_graph || P_layout) ]

Advantages:
- No scaling factor needed (only relative order matters)
- Differentiable
- Clear probabilistic interpretation
"""
import networkx as nx
import torch
import torch.nn.functional as F


class SoftmaxRankingLoss:

    def __init__(
        self,
        device: torch.cuda.device,
        G: nx.Graph = None,
        tau: float = None,
        P_graph: torch.Tensor = None,
    ):
        """
        Initialize the Softmax Ranking Loss calculator.

        Args:
            G: NetworkX graph (if provided, P_graph is computed from the graph)
            device: torch device
            tau: temperature parameter; if None, automatically set to mean(d_graph)
            P_graph: precomputed target distribution [n, n] (used directly if provided)
        """
        self.device = device

        # If precomputed P_graph is provided, use it directly
        if P_graph is not None:
            self.P_graph = P_graph.to(device)
            self.n = P_graph.shape[0]
            self.tau = tau if tau is not None else 1.0
            self.d_graph = None  # not needed
            return

        # Otherwise compute from graph
        if G is None:
            raise ValueError(
                "Must provide either G (graph) or P_graph (precomputed)")

        self.nodes = list(G.nodes())
        self.n = len(self.nodes)

        # Compute graph distances (shortest paths)
        d_graph = torch.zeros((self.n, self.n),
                              dtype=torch.float32,
                              device=device)
        for i, u in enumerate(self.nodes):
            sp_lengths = nx.single_source_shortest_path_length(G, u)
            for v, dist in sp_lengths.items():
                j = self.nodes.index(v)
                d_graph[i, j] = float(dist)
        self.d_graph = d_graph

        # Set temperature parameter
        # Exclude diagonal (distance from node to itself = 0)
        mask = ~torch.eye(self.n, dtype=torch.bool, device=device)
        if tau is None:
            # Adaptive: use the mean of graph distances
            self.tau = d_graph[mask].mean().item()
            if self.tau < 1e-6:
                self.tau = 1.0  # prevent division by zero
        else:
            self.tau = tau

        # Precompute P_graph distribution
        # For each node i, compute P_graph(j|i) = softmax(-d_graph(i,j) / tau)
        # excluding self (j != i)
        self.P_graph = self._compute_softmax_distribution(d_graph)

    def _compute_softmax_distribution(
        self,
        distances: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the softmax distribution for each node.

        Args:
            distances: [n, n] distance matrix

        Returns:
            P: [n, n] probability distribution, P[i,j] = P(j|i), diagonal is 0
        """
        n = distances.shape[0]

        # logits = -d / tau
        logits = -distances / self.tau

        # Set diagonal to -inf (exclude self)
        mask = torch.eye(n, dtype=torch.bool, device=self.device)
        logits = logits.masked_fill(mask, float('-inf'))

        # softmax
        P = F.softmax(logits, dim=1)

        return P

    def _compute_layout_distances(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Compute the Euclidean distance matrix for the layout.

        Args:
            coords: [n, 2] coordinates

        Returns:
            d_layout: [n, n] distance matrix
        """
        # d_layout[i,j] = ||coords[i] - coords[j]||
        squared_norms = (coords**2).sum(dim=1)
        d_sq = squared_norms[:, None] + squared_norms[
            None, :] - 2 * coords @ coords.T
        d_layout = torch.sqrt(torch.clamp(d_sq, min=1e-8))
        return d_layout

    def compute_kl(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Compute KL(P_graph || P_layout).

        Args:
            coords: [n, 2] current layout coordinates

        Returns:
            kl: scalar, mean KL divergence
        """
        # Compute layout distances
        d_layout = self._compute_layout_distances(coords)

        # Compute P_layout
        P_layout = self._compute_softmax_distribution(d_layout)

        # KL(P_graph || P_layout) = sum P_graph * log(P_graph / P_layout)
        # = sum P_graph * (log P_graph - log P_layout)

        # Avoid log(0)
        eps = 1e-8
        log_P_graph = torch.log(self.P_graph + eps)
        log_P_layout = torch.log(P_layout + eps)

        # Diagonal is 0, excluded from computation
        mask = ~torch.eye(self.n, dtype=torch.bool, device=self.device)

        # KL per node
        kl_per_node = (self.P_graph * (log_P_graph - log_P_layout)).sum(dim=1)

        # Average
        kl = kl_per_node.mean()

        return kl

    def __call__(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Compute Softmax Ranking Loss.

        Args:
            coords: [n, 2] coordinates

        Returns:
            loss: KL divergence (lower is better)
        """
        return self.compute_kl(coords)

    def get_tau(self) -> float:
        """Return the temperature parameter."""
        return self.tau
