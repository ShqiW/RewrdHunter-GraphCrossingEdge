"""
Graph Layout Environment for Reinforcement Learning
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from xing import XingLoss


class GraphLayoutEnv(gym.Env):
    """
    Environment for graph layout optimization using RL.

    State: Node coordinates [N x 2] + adjacency info
    Action: (node_index, delta_x, delta_y)
    Reward: Reduction in edge crossings
    """

    def __init__(
        self,
        graph_path: str = None,
        graph: nx.Graph = None,
        max_steps: int = 100,
        patience: int = 10,
        move_scale: float = 0.05,
        initial_layout: str = "neato",
    ):
        """
        Args:
            graph_path: Path to GraphML file
            graph: NetworkX graph (alternative to graph_path)
            max_steps: Maximum steps per episode
            patience: Stop if no improvement for this many steps
            move_scale: Scale factor for node movements (relative to layout size)
            initial_layout: Layout algorithm for initial state ("neato" or "sfdp")
        """
        super().__init__()

        self.max_steps = max_steps
        self.patience = patience
        self.move_scale = move_scale
        self.initial_layout = initial_layout

        # Load graph
        if graph is not None:
            self.graph = graph
        elif graph_path is not None:
            self.graph = nx.read_graphml(graph_path)
            self.graph = nx.convert_node_labels_to_integers(self.graph, ordering="sorted")
        else:
            raise ValueError("Must provide either graph_path or graph")

        self.num_nodes = self.graph.number_of_nodes()
        self.num_edges = self.graph.number_of_edges()

        # Build adjacency matrix
        self.adj_matrix = nx.to_numpy_array(self.graph, dtype=np.float32)

        # Edge list for torch_geometric
        self.edge_index = np.array(list(self.graph.edges())).T
        if len(self.edge_index) == 0:
            self.edge_index = np.zeros((2, 0), dtype=np.int64)

        # Action space: node_index (discrete) + delta_x, delta_y (continuous)
        # We'll use a Dict space
        self.action_space = spaces.Dict({
            "node": spaces.Discrete(self.num_nodes),
            "delta": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        })

        # Observation space: node coordinates
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_nodes, 2), dtype=np.float32
        )

        # Initialize crossing loss calculator
        self.xing_loss = XingLoss(self.graph, soft=False)

        # State variables
        self.coords = None
        self.current_crossings = None
        self.best_crossings = None
        self.steps = 0
        self.no_improve_steps = 0
        self.layout_scale = 1.0

    def _compute_crossings(self, coords: np.ndarray) -> int:
        """Compute number of edge crossings."""
        coords_tensor = torch.tensor(coords, dtype=torch.float32)
        crossings = self.xing_loss(coords_tensor)
        return int(crossings.item())

    def _get_initial_layout(self) -> np.ndarray:
        """Get initial layout using graphviz."""
        pos = nx.nx_agraph.graphviz_layout(self.graph, prog=self.initial_layout)
        coords = np.array([[pos[v][0], pos[v][1]] for v in self.graph.nodes()], dtype=np.float32)
        return coords

    def _normalize_coords(self, coords: np.ndarray) -> np.ndarray:
        """Normalize coordinates to [0, 1] range."""
        min_coords = coords.min(axis=0)
        max_coords = coords.max(axis=0)
        scale = max_coords - min_coords
        scale[scale == 0] = 1.0  # Avoid division by zero
        self.layout_scale = scale.mean()
        return (coords - min_coords) / scale

    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)

        # Get initial layout
        self.coords = self._get_initial_layout()
        self.coords = self._normalize_coords(self.coords)

        # Compute initial crossings
        self.current_crossings = self._compute_crossings(self.coords)
        self.best_crossings = self.current_crossings
        self.initial_crossings = self.current_crossings

        # Reset counters
        self.steps = 0
        self.no_improve_steps = 0

        info = {
            "crossings": self.current_crossings,
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
        }

        return self.coords.copy(), info

    def step(self, action):
        """
        Execute one step.

        Args:
            action: Dict with "node" (int) and "delta" (np.array of shape [2])

        Returns:
            observation, reward, terminated, truncated, info
        """
        node_idx = action["node"]
        delta = action["delta"] * self.move_scale

        # Move the selected node
        self.coords[node_idx] += delta

        # Compute new crossings
        new_crossings = self._compute_crossings(self.coords)

        # Reward = reduction in crossings
        reward = self.current_crossings - new_crossings

        # Update state
        self.current_crossings = new_crossings
        self.steps += 1

        # Track best and patience
        if new_crossings < self.best_crossings:
            self.best_crossings = new_crossings
            self.no_improve_steps = 0
        else:
            self.no_improve_steps += 1

        # Check termination
        terminated = False
        truncated = False

        if self.current_crossings == 0:
            terminated = True  # Perfect layout
        elif self.steps >= self.max_steps:
            truncated = True  # Max steps reached
        elif self.no_improve_steps >= self.patience:
            truncated = True  # No improvement

        info = {
            "crossings": self.current_crossings,
            "best_crossings": self.best_crossings,
            "initial_crossings": self.initial_crossings,
            "improvement": self.initial_crossings - self.best_crossings,
            "steps": self.steps,
        }

        return self.coords.copy(), reward, terminated, truncated, info

    def get_graph_data(self):
        """Return graph data for GNN (edge_index, num_nodes)."""
        return {
            "edge_index": torch.tensor(self.edge_index, dtype=torch.long),
            "num_nodes": self.num_nodes,
            "adj_matrix": torch.tensor(self.adj_matrix, dtype=torch.float32),
        }


class MultiGraphEnv(gym.Env):
    """
    Environment that samples from multiple graphs for training.
    """

    def __init__(
        self,
        graph_list_file: str,
        data_dir: str,
        max_steps: int = 100,
        patience: int = 10,
        move_scale: float = 0.05,
        initial_layout: str = "neato",
    ):
        """
        Args:
            graph_list_file: Path to file containing list of graph paths
            data_dir: Base directory for graph files
            max_steps: Maximum steps per episode
            patience: Stop if no improvement for this many steps
            move_scale: Scale factor for node movements
            initial_layout: Layout algorithm for initial state
        """
        super().__init__()

        self.data_dir = Path(data_dir)
        self.max_steps = max_steps
        self.patience = patience
        self.move_scale = move_scale
        self.initial_layout = initial_layout

        # Load graph list
        with open(graph_list_file, 'r') as f:
            self.graph_files = [line.strip() for line in f if line.strip()]

        print(f"Loaded {len(self.graph_files)} graphs for training")

        # Create initial env with first graph to get spaces
        self._current_env = None
        self._sample_new_graph()

        # Use variable-size observation (will handle in wrapper)
        self.observation_space = self._current_env.observation_space
        self.action_space = self._current_env.action_space

    def _sample_new_graph(self):
        """Sample a new graph and create environment."""
        graph_file = np.random.choice(self.graph_files)
        graph_path = self.data_dir / graph_file

        self._current_env = GraphLayoutEnv(
            graph_path=str(graph_path),
            max_steps=self.max_steps,
            patience=self.patience,
            move_scale=self.move_scale,
            initial_layout=self.initial_layout,
        )

        # Update spaces for new graph size
        self.observation_space = self._current_env.observation_space
        self.action_space = self._current_env.action_space

    def reset(self, seed=None, options=None):
        """Reset with a new random graph."""
        self._sample_new_graph()
        return self._current_env.reset(seed=seed, options=options)

    def step(self, action):
        """Execute step on current graph."""
        return self._current_env.step(action)

    def get_graph_data(self):
        """Return current graph data for GNN."""
        return self._current_env.get_graph_data()

    @property
    def num_nodes(self):
        return self._current_env.num_nodes
