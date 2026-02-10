import networkx as nx
import torch

class StressLoss:
    def __init__(self, G: nx.Graph, device=None, soft = True):
        """
        Precompute graph distances once and store as a tensor.
        G: NetworkX graph
        device: torch device (cpu or cuda)
        """
        if device is None:
            device = torch.device("cpu")
        self.device = device

        self.nodes = list(G.nodes())
        self.n = len(self.nodes)
        self.soft = soft # not used for now

        # Compute graph distances once
        d_graph = torch.zeros((self.n, self.n), dtype=torch.float32, device=device)
        for i, u in enumerate(self.nodes):
            sp_lengths = nx.single_source_shortest_path_length(G, u)
            for v, dist in sp_lengths.items():
                j = self.nodes.index(v)
                d_graph[i, j] = float(dist)
        self.d_graph = d_graph

        # Mask to exclude i<=j
        #self.mask = ~torch.eye(self.n, dtype=torch.bool, device=device)
        self.mask = torch.triu(torch.ones(self.n, self.n, dtype=torch.bool, device=device), diagonal=1)

        
    def calculate_scale_factor(self, coords: torch.Tensor):
        squared_norms = (coords**2).sum(dim=1)
        d_embed_sq = squared_norms[:, None] + squared_norms[None, :] - 2 * coords @ coords.T
        #d_embed = torch.sqrt(torch.clamp(d_embed_sq, min=0.0))
        d_embed = torch.sqrt(torch.clamp(d_embed_sq, min=0.0) + 1e-3) #
        
        # Mask out zeros
        valid_mask = self.mask & (self.d_graph > 0)
        W = 1.0 / torch.clamp(self.d_graph[valid_mask] ** 2, min=1e-1)
        D_ij = self.d_graph[valid_mask]
        D_proj = d_embed[valid_mask]
    
        numerator = (W * D_proj * D_ij).sum()
        denominator = (W * D_proj**2).sum()
        alpha = numerator / denominator if denominator > 0 else torch.tensor(1.0, device=self.device)
#        alpha = torch.clamp(numerator / (denominator + 1e-6), max=10.0) if denominator > 0 else torch.tensor(1.0, device=self.device)
        return alpha
    
    def __call__(self, coords: torch.Tensor):
        alpha = self.calculate_scale_factor(coords)
        coords_scaled = alpha * coords
    
        squared_norms = (coords_scaled**2).sum(dim=1)
        d_embed_sq = squared_norms[:, None] + squared_norms[None, :] - 2 * coords_scaled @ coords_scaled.T
#        d_embed = torch.sqrt(torch.clamp(d_embed_sq, min=0.0))
        d_embed = torch.sqrt(torch.clamp(d_embed_sq, min=0.0) + 1e-3)  

        valid_mask = self.mask & (self.d_graph > 0)
        W = 1.0 / torch.clamp(self.d_graph[valid_mask] ** 2, min=1e-3)
        stress = ((d_embed - self.d_graph)**2)[valid_mask] * W
        k = self.n*(self.n - 1) /2
        return stress.sum()/k