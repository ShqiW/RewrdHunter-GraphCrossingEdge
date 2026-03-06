"""
Discrete action space graph layout environment.

Action space: Discrete(num_nodes * 8)
Action encoding: action = node_id * 8 + direction_id
8 directions: ↑ ↗ → ↘ ↓ ↙ ← ↖

State Space (GAT features):
- Node features: [x, y, degree]
- Edge features: [edge_length, is_crossing]
"""
from dataclasses import dataclass
from typing import Optional
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
import torch
from pathlib import Path

from src.tasks.base import BaseEnvConfig
from src.losses.xing import XingLoss


# 8 direction unit vectors
DIRECTIONS = np.array([
    [0, 1],    # 0: ↑
    [1, 1],    # 1: ↗
    [1, 0],    # 2: →
    [1, -1],   # 3: ↘
    [0, -1],   # 4: ↓
    [-1, -1],  # 5: ↙
    [-1, 0],   # 6: ←
    [-1, 1],   # 7: ↖
], dtype=np.float32)

# Normalize direction vectors
DIRECTIONS = DIRECTIONS / np.linalg.norm(DIRECTIONS, axis=1, keepdims=True)

NUM_DIRECTIONS = 8


@dataclass
class DiscreteEnvConfig(BaseEnvConfig):
    """Discrete environment configuration"""
    type: str = "discrete"

    # Reward weights (R = w1 * ΔR_cross + w2 * ΔR_structure)
    crossing_weight: float = 1.0
    structure_weight: float = 0.1

    # Structure consistency method: "stress" | "rank" | "softmax"
    structure_method: str = "softmax"

    # Method-specific parameters
    softmax_tau: Optional[float] = None  # None = adaptive mean(d_graph)

    use_potential_shaping: bool = True

    # Crossing computation
    soft_crossing: bool = True


class DiscreteGraphEnv(gym.Env):
    """
    Discrete action space graph layout environment.

    Action space: Discrete(num_nodes * 8)
    Action decoding: action = node_id * 8 + direction_id
    """

    def __init__(
        self,
        graph: nx.Graph = None,
        graph_path: str = None,
        graph_data=None,  # GraphData from RomeDataset
        config: DiscreteEnvConfig = None,
        **kwargs,
    ):
        super().__init__()

        # Use config or kwargs
        if config is None:
            config = DiscreteEnvConfig(**kwargs)

        self.config = config
        self.max_steps = config.max_steps
        self.patience = config.patience
        self.move_scale = config.move_scale
        self.initial_layout = config.initial_layout
        self.crossing_weight = config.crossing_weight
        self.structure_weight = config.structure_weight
        self.use_potential_shaping = config.use_potential_shaping

        # Load graph from various sources
        if graph_data is not None:
            # From RomeDataset GraphData
            self.num_nodes = graph_data.num_nodes
            self.edge_index = graph_data.edge_index.numpy()
            self.graph_name = graph_data.graph_name

            # Build nx.Graph for XingLoss
            self.graph = nx.Graph()
            self.graph.add_nodes_from(range(self.num_nodes))
            edges = self.edge_index.T[:self.edge_index.shape[1] // 2]  # Remove duplicates
            self.graph.add_edges_from(edges.tolist())

        elif graph is not None:
            self.graph = graph
            self.num_nodes = self.graph.number_of_nodes()
            self.graph_name = None

        elif graph_path is not None:
            self.graph = nx.read_graphml(graph_path)
            self.graph = nx.convert_node_labels_to_integers(self.graph, ordering="sorted")
            self.num_nodes = self.graph.number_of_nodes()
            self.graph_name = graph_path

        else:
            raise ValueError("Must provide graph, graph_path, or graph_data")

        self.num_edges = self.graph.number_of_edges()

        # Adjacency matrix and edge list (from nx.Graph)
        self.adj_matrix = nx.to_numpy_array(self.graph, dtype=np.float32)
        if graph_data is None:
            edges = list(self.graph.edges())
            if len(edges) > 0:
                edge_arr = np.array(edges, dtype=np.int64).T
                # Make undirected (both directions)
                self.edge_index = np.concatenate([edge_arr, edge_arr[::-1]], axis=1)
            else:
                self.edge_index = np.zeros((2, 0), dtype=np.int64)

        # Node degree (fixed throughout training)
        self.degree = np.array([self.graph.degree(i) for i in range(self.num_nodes)], dtype=np.float32)

        # Discrete action space
        self.num_actions = self.num_nodes * NUM_DIRECTIONS
        self.action_space = spaces.Discrete(self.num_actions)

        # Observation space: node features [x, y, degree]
        # Note: actual observation includes dynamic edge features computed separately
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_nodes, 3), dtype=np.float32
        )

        # Loss calculators
        self.xing_loss = XingLoss(self.graph, soft=config.soft_crossing)

        # Structure loss (select method based on config)
        match config.structure_method:
            case "stress":
                from src.losses.stress import StressLoss
                self.structure_loss = StressLoss(self.graph)
            case "rank":
                from src.losses.rank_matching import RankMatchingLoss
                self.structure_loss = RankMatchingLoss(self.graph, tau=config.softmax_tau)
            case "softmax":
                from src.losses.softmax_ranking import SoftmaxRankingLoss
                self.structure_loss = SoftmaxRankingLoss(self.graph, tau=config.softmax_tau)
            case _:
                raise ValueError(f"Unknown structure method: {config.structure_method}")

        # State variables
        self.coords = None
        self.current_crossings = None
        self.current_structure = None
        self.best_crossings = None
        self.steps = 0
        self.no_improve_steps = 0

    def _decode_action(self, action: int):
        """Decode action: action -> (node_id, direction_id)"""
        node_id = action // NUM_DIRECTIONS
        direction_id = action % NUM_DIRECTIONS
        return node_id, direction_id

    def _compute_crossings(self, coords: np.ndarray) -> float:
        coords_tensor = torch.tensor(coords, dtype=torch.float32)
        return self.xing_loss(coords_tensor).item()

    def _compute_structure(self, coords: np.ndarray) -> float:
        """Compute structure loss (edge length + overlap)"""
        coords_tensor = torch.tensor(coords, dtype=torch.float32)
        return self.structure_loss(coords_tensor).item()

    def _compute_potential(self, coords: np.ndarray) -> float:
        """Potential function for shaping: -(crossing + w2 * structure)"""
        crossing = self._compute_crossings(coords)
        structure = self._compute_structure(coords)
        return -(crossing + self.structure_weight * structure)

    def _compute_edge_lengths(self, coords: np.ndarray) -> np.ndarray:
        """Compute edge lengths for edge features"""
        if self.edge_index.shape[1] == 0:
            return np.array([], dtype=np.float32)

        src = self.edge_index[0]
        dst = self.edge_index[1]
        diff = coords[src] - coords[dst]
        lengths = np.linalg.norm(diff, axis=1).astype(np.float32)
        return lengths

    def _compute_edge_crossings(self, coords: np.ndarray) -> np.ndarray:
        """
        Compute which edges are crossing (0 or 1 per edge).
        Returns binary array of shape [num_edges].
        """
        num_edges_undirected = self.edge_index.shape[1] // 2
        if num_edges_undirected == 0:
            return np.array([], dtype=np.float32)

        # Get undirected edges (first half)
        edges = self.edge_index[:, :num_edges_undirected].T

        # Check each pair of edges for crossing
        is_crossing = np.zeros(num_edges_undirected, dtype=np.float32)

        for i, (u1, v1) in enumerate(edges):
            p1, p2 = coords[u1], coords[v1]
            for j, (u2, v2) in enumerate(edges[i+1:], i+1):
                # Skip adjacent edges
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    continue
                p3, p4 = coords[u2], coords[v2]
                if self._segments_intersect(p1, p2, p3, p4):
                    is_crossing[i] = 1.0
                    is_crossing[j] = 1.0

        # Duplicate for both directions
        return np.concatenate([is_crossing, is_crossing])

    def _segments_intersect(self, p1, p2, p3, p4) -> bool:
        """Check if line segment p1-p2 intersects p3-p4"""
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

        return (ccw(p1,p3,p4) != ccw(p2,p3,p4)) and (ccw(p1,p2,p3) != ccw(p1,p2,p4))

    def _get_node_features(self, coords: np.ndarray) -> np.ndarray:
        """
        Get node features: [x, y, degree]
        Shape: [num_nodes, 3]
        """
        features = np.column_stack([
            coords,
            self.degree.reshape(-1, 1)
        ]).astype(np.float32)
        return features

    def _get_edge_features(self, coords: np.ndarray) -> np.ndarray:
        """
        Get edge features: [edge_length, is_crossing]
        Shape: [num_edges, 2]
        """
        edge_lengths = self._compute_edge_lengths(coords)
        is_crossing = self._compute_edge_crossings(coords)

        if len(edge_lengths) == 0:
            return np.zeros((0, 2), dtype=np.float32)

        features = np.column_stack([
            edge_lengths,
            is_crossing
        ]).astype(np.float32)
        return features

    def _get_initial_layout(self) -> np.ndarray:
        if self.initial_layout == "random":
            coords = np.random.rand(self.num_nodes, 2).astype(np.float32)
        elif self.initial_layout == "neato":
            pos = nx.nx_agraph.graphviz_layout(self.graph, prog="neato")
            coords = np.array([[pos[v][0], pos[v][1]] for v in self.graph.nodes()], dtype=np.float32)
        elif self.initial_layout == "sfdp":
            pos = nx.nx_agraph.graphviz_layout(self.graph, prog="sfdp")
            coords = np.array([[pos[v][0], pos[v][1]] for v in self.graph.nodes()], dtype=np.float32)
        else:
            pos = nx.spring_layout(self.graph)
            coords = np.array([[pos[v][0], pos[v][1]] for v in self.graph.nodes()], dtype=np.float32)

        return self._normalize_coords(coords)

    def _normalize_coords(self, coords: np.ndarray) -> np.ndarray:
        min_coords = coords.min(axis=0)
        max_coords = coords.max(axis=0)
        scale = max_coords - min_coords
        scale[scale == 0] = 1.0
        return (coords - min_coords) / scale

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.coords = self._get_initial_layout()
        self.current_crossings = self._compute_crossings(self.coords)
        self.current_structure = self._compute_structure(self.coords)
        self.current_potential = self._compute_potential(self.coords)

        self.initial_crossings = self.current_crossings
        self.initial_structure = self.current_structure
        self.best_crossings = self.current_crossings

        self.steps = 0
        self.no_improve_steps = 0

        # Return node features as observation
        node_features = self._get_node_features(self.coords)

        info = {
            "crossings": self.current_crossings,
            "structure_loss": self.current_structure,
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
        }

        return node_features, info

    def step(self, action: int):
        """Execute discrete action"""
        # Decode action
        node_id, direction_id = self._decode_action(action)

        # Get movement vector
        delta = DIRECTIONS[direction_id] * self.move_scale

        # Save old state
        old_potential = self.current_potential

        # Move node
        self.coords[node_id] += delta

        # Clip to [0, 1]
        self.coords = np.clip(self.coords, 0, 1)

        # Compute new state
        new_crossings = self._compute_crossings(self.coords)
        new_structure = self._compute_structure(self.coords)
        new_potential = self._compute_potential(self.coords)

        # Compute reward: R = w1 * ΔR_cross + w2 * ΔR_structure
        crossing_reward = self.current_crossings - new_crossings
        structure_reward = self.current_structure - new_structure

        if self.use_potential_shaping:
            gamma = 0.99
            potential_shaping = gamma * new_potential - old_potential
            reward = (
                self.crossing_weight * crossing_reward +
                self.structure_weight * structure_reward +
                0.5 * potential_shaping
            )
        else:
            reward = (
                self.crossing_weight * crossing_reward +
                self.structure_weight * structure_reward
            )

        # Update state
        self.current_crossings = new_crossings
        self.current_structure = new_structure
        self.current_potential = new_potential
        self.steps += 1

        # Track best
        if new_crossings < self.best_crossings:
            self.best_crossings = new_crossings
            self.no_improve_steps = 0
        else:
            self.no_improve_steps += 1

        # Termination
        terminated = False
        truncated = False

        if self.current_crossings == 0:
            terminated = True
            reward += 10.0  # Bonus for perfect layout
        elif self.steps >= self.max_steps:
            truncated = True
        elif self.no_improve_steps >= self.patience:
            truncated = True

        # Return node features as observation
        node_features = self._get_node_features(self.coords)

        info = {
            "crossings": self.current_crossings,
            "structure_loss": self.current_structure,
            "best_crossings": self.best_crossings,
            "initial_crossings": self.initial_crossings,
            "initial_structure": self.initial_structure,
            "improvement": self.initial_crossings - self.current_crossings,
            "structure_improvement": self.initial_structure - self.current_structure,
            "steps": self.steps,
            "action_node": node_id,
            "action_direction": direction_id,
        }

        return node_features, reward, terminated, truncated, info

    def get_graph_data(self):
        """Return graph data for GAT"""
        edge_features = self._get_edge_features(self.coords)
        return {
            "edge_index": torch.tensor(self.edge_index, dtype=torch.long),
            "edge_attr": torch.tensor(edge_features, dtype=torch.float32),
            "num_nodes": self.num_nodes,
            "degree": torch.tensor(self.degree, dtype=torch.float32),
        }

    def get_coords(self) -> np.ndarray:
        """Return current coordinates"""
        return self.coords.copy()
