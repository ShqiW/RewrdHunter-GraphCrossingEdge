"""
Rome Graph Dataset for Graph Layout Optimization.

Uses per-graph .pt files for efficient storage of variable-size graphs.
Precomputes static features including P_graph for softmax ranking.
"""
import os
import torch
import torch.nn.functional as F
import networkx as nx
from pathlib import Path
from tqdm import tqdm
from typing import Optional, List

from torch.utils.data import Dataset
from src.data.data import GraphData


class RomeDataset(Dataset):
    """
    Rome graph dataset with precomputed features for softmax ranking.

    Each graph contains:
        - edge_index: [2, num_edges] graph topology
        - num_nodes: int
        - degree: [num_nodes] node degrees
        - graph_distance: [num_nodes, num_nodes] shortest path distances
        - P_graph: [num_nodes, num_nodes] softmax target distribution
        - tau: float, temperature parameter

    Args:
        root: Root directory (should contain 'rome/' subdir and txt files)
        split: 'train' or 'test'
        force_reload: If True, reprocess even if cache exists
    """

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        force_reload: bool = False,
    ):
        self.root = Path(root)
        self.split = split
        self.processed_dir = self.root / "processed" / split

        # Read graph list
        list_file = self.root / f"{split}_graph.txt"
        with open(list_file, "r") as f:
            self.graph_files = [line.strip() for line in f if line.strip()]

        # Process if needed
        if force_reload or not self._is_processed():
            self._process()

        # Load index
        self.index = self._load_index()
        self._cache: dict = {}

    def _is_processed(self) -> bool:
        """Check if dataset is already processed."""
        index_file = self.processed_dir / "index.pt"
        return index_file.exists()

    def _load_index(self) -> List[str]:
        """Load index of processed graphs."""
        index_file = self.processed_dir / "index.pt"
        return torch.load(index_file)

    def _process(self):
        from src.plot import plot_sequential, build_nx_graph
        """Process all graphs and save to disk."""
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        processed_names = []
        print(
            f"Processing {self.split} dataset ({len(self.graph_files)} graphs)..."
        )

        for graph_file in tqdm(self.graph_files,
                               desc=f"Processing {self.split}"):
            # Extract filename
            filename = os.path.basename(graph_file)
            full_path = self.root / "rome" / filename

            if not full_path.exists():
                continue

            data = self._process_single_graph(str(full_path), filename)
            if data is None:
                continue
            graphdata = GraphData(**data)

            # plot_path = plot_sequential(
            #     [{
            #         "graph": build_nx_graph(graphdata),
            #         "coords": graphdata.neato_coords.numpy(),
            #         "graph_name": graphdata.graph_name,
            #         "best_xing": graphdata.neato_xing,
            #     }],
            #     out_dir=self.processed_dir,
            # )
            # print(plot_path)
            # Save to individual file
            save_path = self.processed_dir / f"{filename}.pt"
            torch.save(data, save_path)
            processed_names.append(filename)

        # Save index
        index_file = self.processed_dir / "index.pt"
        torch.save(processed_names, index_file)
        print(f"Processed {len(processed_names)} graphs")

    def _process_single_graph(
        self,
        graph_path: str,
        graph_name: str,
    ) -> Optional[dict]:
        """Process a single graph file."""
        # Load graph
        G = nx.read_graphml(graph_path)
        G = nx.convert_node_labels_to_integers(G, ordering="sorted")

        # Skip empty or trivial graphs
        if G.number_of_nodes() < 3 or G.number_of_edges() < 2:
            return None

        # Ensure connected
        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()
            G = nx.convert_node_labels_to_integers(G, ordering="sorted")

        num_nodes = G.number_of_nodes()

        # Edge index
        edges = list(G.edges())
        if len(edges) > 0:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            # Make undirected
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        # Node degree
        degree = torch.tensor([G.degree(i) for i in range(num_nodes)],
                              dtype=torch.float32)

        # Graph distance (shortest path)
        graph_distance = self._compute_graph_distance(G)

        # P_graph and tau for softmax ranking
        P_graph, tau = self._compute_softmax_target(graph_distance)

        # Neato layout (raw, unnormalized) — precomputed once for reuse
        neato_coords = self._compute_neato_coords(G)
        neato_xing = self._compute_crossings(G, neato_coords)

        return {
            "edge_index": edge_index,
            "num_nodes": num_nodes,
            "degree": degree,
            "graph_distance": graph_distance,
            "P_graph": P_graph,
            "tau": tau,
            "graph_name": graph_name,
            "neato_coords": neato_coords,
            "neato_xing": neato_xing,
        }

    def _compute_neato_coords(self, G: nx.Graph) -> torch.Tensor:
        """Compute neato layout and return raw (unnormalized) coordinates."""
        pos = nx.nx_agraph.graphviz_layout(G, prog="neato")
        # pos = nx.nx_agraph.graphviz_layout(G, prog="sfdp")
        n = G.number_of_nodes()
        coords = torch.tensor(
            [[pos[v][0], pos[v][1]] for v in range(n)],
            dtype=torch.float32,
        )
        return coords

    def _compute_crossings(self, G: nx.Graph, coords: torch.Tensor) -> int:
        """Count hard edge crossings for a given layout."""
        from src.plot import count_hard_crossings
        return count_hard_crossings(G, coords.numpy(), device="cuda")

    def _compute_graph_distance(self, G: nx.Graph) -> torch.Tensor:
        """Compute shortest path distance matrix."""
        n = G.number_of_nodes()

        d_graph = torch.zeros((n, n), dtype=torch.float32)

        for u in range(n):
            sp_lengths = nx.single_source_shortest_path_length(G, u)
            for v, dist in sp_lengths.items():
                d_graph[u, v] = float(dist)

        return d_graph

    def _compute_softmax_target(self, graph_distance: torch.Tensor):
        """
        Compute softmax target distribution P_graph and temperature tau.
        """
        n = graph_distance.shape[0]

        # Compute tau (excluding diagonal)
        mask = ~torch.eye(n, dtype=torch.bool)
        tau = graph_distance[mask].mean().item()
        if tau < 1e-6:
            tau = 1.0

        # Compute P_graph
        logits = -graph_distance / tau

        # Set diagonal to -inf (exclude self)
        diag_mask = torch.eye(n, dtype=torch.bool)
        logits = logits.masked_fill(diag_mask, float('-inf'))

        # Softmax
        P_graph = F.softmax(logits, dim=1)

        return P_graph, tau

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> GraphData:
        """Load a graph by index (cached in memory after first access)."""
        if idx not in self._cache:
            graph_name = self.index[idx]
            data_path = self.processed_dir / f"{graph_name}.pt"
            self._cache[idx] = torch.load(data_path)
        data_dict = self._cache[idx]

        return GraphData(
            edge_index=data_dict["edge_index"],
            num_nodes=data_dict["num_nodes"],
            degree=data_dict["degree"],
            graph_distance=data_dict["graph_distance"],
            P_graph=data_dict["P_graph"],
            tau=data_dict["tau"],
            graph_name=data_dict["graph_name"],
            neato_coords=data_dict["neato_coords"],
            neato_xing=data_dict["neato_xing"],
        )

    def sample(self) -> GraphData:
        """Random sample a graph."""
        idx = torch.randint(0, len(self), (1, )).item()
        return self[idx]


if __name__ == "__main__":
    print("Loading train dataset...")
    train_dataset = RomeDataset(root="data", split="train")
    print(f"Train dataset: {len(train_dataset)} graphs")

    print("\nLoading test dataset...")
    test_dataset = RomeDataset(root="data", split="test")
    print(f"Test dataset: {len(test_dataset)} graphs")

    # Check a sample
    sample = train_dataset[0]
    print(f"\nSample graph: {sample.graph_name}")
    print(f"  Nodes: {sample.num_nodes}")
    print(f"  Edges: {sample.edge_index.shape[1] // 2}")
    print(f"  P_graph shape: {sample.P_graph.shape}")
    print(f"  P_graph row sum: {sample.P_graph.sum(dim=1)[:3]}")
    print(f"  Tau: {sample.tau:.4f}")
