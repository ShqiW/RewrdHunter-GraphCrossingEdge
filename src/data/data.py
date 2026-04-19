from dataclasses import dataclass

import torch


@dataclass
class GraphData:
    """Data container for a single graph."""
    edge_index: torch.Tensor  # [2, num_edges]
    num_nodes: int
    degree: torch.Tensor  # [num_nodes]
    graph_distance: torch.Tensor  # [num_nodes, num_nodes]
    P_graph: torch.Tensor  # [num_nodes, num_nodes]
    tau: float
    graph_name: str
    neato_coords: torch.Tensor  # [num_nodes, 2] raw neato layout (unnormalized)
    neato_xing: int  # crossing count on raw neato layout
    node_ids: list = None  # original node labels (e.g. ["n0","n1","n10",...]) in integer-index order
