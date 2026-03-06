import torch
import networkx as nx

class XingLoss:
    def __init__(self, G: nx.Graph, device = None, soft = False, sharpness=10.0):
        # Store edges as a long tensor [num_edges, 2]
        nodes = list(G.nodes())
        edges = [[nodes.index(i), nodes.index(j)] for i,j in G.edges]
        self.edges = torch.tensor(edges, dtype=torch.long)
        if device is None:
            device = torch.device("cpu")
        self.device = device
        self.soft = soft
        self.sharpness = sharpness
        
    @staticmethod
    def cross_2d(v, u):
        return v[..., 0] * u[..., 1] - v[..., 1] * u[..., 0]

    @staticmethod
    def dot_2d(v, u):
        return torch.sum(v * u, dim=-1)
    
    @staticmethod
    def tent_function(x, y):
        # Compute linear drop-off from center
        val = 1 - 2*torch.abs(x - 0.5) - 2*torch.abs(y - 0.5)
        # Clamp to zero outside [0,1]^2 and negative values
        val = torch.clamp(val, min=0.0)
        return val # the voumn of the prymid is 1/3. We reply on learning rate to control instead pf rescaling by *3
    
    @staticmethod
    def gaussian_tent(x, y, sigma=0.15):
        """
        Smoothed tent function using a 2D Gaussian
        """
        dist2 = (x - 0.5)**2 + (y - 0.5)**2
        val = torch.exp(-dist2 / (2 * sigma**2))
        val = val * ((x >= 0) & (x <= 1) & (y >= 0) & (y <= 1))
        return val

    def edges_intersect(self, edge_1_start_pos, edge_1_end_pos,
                        edge_2_start_pos, edge_2_end_pos, eps=1e-6):
        p = edge_1_start_pos
        q = edge_2_start_pos
        r = edge_1_end_pos - p
        s = edge_2_end_pos - q

        p = p + eps * r
        q = q + eps * s
        r = r * (1 - 2 * eps)
        s = s * (1 - 2 * eps)

        qmp = q - p
        qmpxs = XingLoss.cross_2d(qmp, s)
        qmpxr = XingLoss.cross_2d(qmp, r)
        rxs = XingLoss.cross_2d(r, s)
        rdr = XingLoss.dot_2d(r, r)

        t0 = XingLoss.dot_2d(qmp, r) / (rdr + 1e-12)
        t1 = t0 + XingLoss.dot_2d(s, r) / (rdr + 1e-12)
    

        # t: how far along edge 1 the intersection point is. 0=>at start of edge 1, 1=>at end of edge 1
        t = qmpxs / (rxs + 1e-12)
        u = qmpxr / (rxs + 1e-12)

        def sigmoid(x):
            return torch.sigmoid(x*self.sharpness)
        
        if self.soft:
            # --- Soft indicator using sigmoid ---
            p5 = torch.tensor(0.5, device=rxs.device, dtype=rxs.dtype)
            M_peak = sigmoid(p5) * (1 - sigmoid(-p5))   # ≈ 0.387
            def inside_norm(t):
                M = sigmoid(t) * (1 - sigmoid(t - 1))
                return M / (M_peak + 1e-12)

            Mt = inside_norm(t)
            Mu = inside_norm(u)
            P_cross = Mt * Mu # this function is ~1 when t and u in [0,1], as low as 0.3 when t or u is far outside [0,1]

            return (P_cross) 
            #return XingLoss.tent_function(t, u)  # triangular drop-off from center
            #return XingLoss.gaussian_tent(t, u)  # triangular drop-off from center
        
        else:
            zero = torch.tensor(0.0, device=rxs.device, dtype=rxs.dtype)
            parallel = torch.isclose(rxs, zero)
            collinear = parallel & torch.isclose(qmpxr, zero)
            intersects_collinear = collinear & ((torch.max(t0, t1) > 0) & (torch.min(t0, t1) < 1))
            intersects_skew = ~parallel & (t > 0) & (t < 1) & (u > 0) & (u < 1)

            return intersects_collinear | intersects_skew

    
    def __call__(self, coords: torch.Tensor) -> torch.Tensor:
        """
        coords: Tensor of shape [num_nodes, >=2], returns scalar total crossings
        """
        num_edges = self.edges.shape[0]
        idx_i, idx_j = torch.triu_indices(num_edges, num_edges, offset=1)
        edge_i = self.edges[idx_i]
        edge_j = self.edges[idx_j]

        # Remove pairs sharing a node
        no_shared_nodes = ~(
            (edge_i[:, 0] == edge_j[:, 0]) |
            (edge_i[:, 0] == edge_j[:, 1]) |
            (edge_i[:, 1] == edge_j[:, 0]) |
            (edge_i[:, 1] == edge_j[:, 1])
        )
        edge_i = edge_i[no_shared_nodes]
        edge_j = edge_j[no_shared_nodes]

        if edge_i.shape[0] == 0:
            return torch.tensor(0.0, device=coords.device)

        # Get endpoints
        edge_1_start_pos = coords[edge_i[:, 0], :2]
        edge_1_end_pos   = coords[edge_i[:, 1], :2]
        edge_2_start_pos = coords[edge_j[:, 0], :2]
        edge_2_end_pos   = coords[edge_j[:, 1], :2]

        crossings = self.edges_intersect(edge_1_start_pos, edge_1_end_pos,
                                         edge_2_start_pos, edge_2_end_pos)
        return crossings.sum().float()
