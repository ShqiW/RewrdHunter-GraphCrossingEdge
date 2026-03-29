"""
Shared base class for GAT-policy graph layout environments.

DiscreteGraphEnv and ContinuousGraphEnv differ only in action space and
step(); everything else lives here to avoid duplication.
"""
from __future__ import annotations

from abc import abstractmethod

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Batch, Data  # type: ignore[import]

from src.data.data import GraphData
from src.envs.base import BaseGraphEnv
from src.envs.graph_layout_state import GraphLayoutState
from src.envs.utils import get_initial_layout
from src.tasks.base import BaseEnvConfig


class GATGraphEnv(BaseGraphEnv):
    """
    Abstract base for GAT-based graph layout environments.

    Concrete subclasses must define ``action_space``, ``observation_space``,
    and implement ``step()``.  All shared setup, feature helpers, reset, and
    policy-interface methods are provided here.
    """

    def __init__(
        self,
        graph_data: GraphData,
        device,
        config: BaseEnvConfig,
    ) -> None:
        super().__init__()

        self.config = config
        self.device = device
        self.move_scale: float = config.move_scale
        self.initial_layout: str = config.initial_layout
        self.crossing_weight: float = getattr(config, "crossing_weight", 1.0)
        self.structure_weight: float = getattr(config, "structure_weight", 0.3)
        self.use_potential_shaping: bool = getattr(
            config, "use_potential_shaping", False
        )

        # ── Graph topology ─────────────────────────────────────────────────────
        self.num_nodes: int = graph_data.num_nodes
        self.edge_index: np.ndarray = graph_data.edge_index.numpy()
        self.graph_name: str | None = graph_data.graph_name
        self.neato_coords: np.ndarray = graph_data.neato_coords.numpy()

        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(self.num_nodes))
        self.graph.add_edges_from(
            self.edge_index.T[: self.edge_index.shape[1] // 2].tolist()
        )
        self.num_edges: int = self.graph.number_of_edges()
        # C(E,2) = max possible crossing pairs; used to normalise crossing reward
        self.max_crossings: float = max(1.0, self.num_edges * (self.num_edges - 1) / 2)
        self.degree: np.ndarray = np.array(
            [self.graph.degree(i) for i in range(self.num_nodes)],
            dtype=np.float32,
        )

        # ── Structure loss ─────────────────────────────────────────────────────
        structure_method: str = getattr(config, "structure_method", "softmax")
        softmax_tau = getattr(config, "softmax_tau", 1)
        match structure_method:
            case "stress":
                from src.losses.stress import StressLoss
                self.structure_loss = StressLoss(self.graph, device=device)
            case "rank":
                from src.losses.rank_matching import RankMatchingLoss
                self.structure_loss = RankMatchingLoss(
                    self.graph,
                    tau=softmax_tau or 1.0,
                    device=device,
                )
            case "softmax":
                from src.losses.softmax_ranking import SoftmaxRankingLoss
                self.structure_loss = SoftmaxRankingLoss(
                    P_graph=graph_data.P_graph,
                    tau=graph_data.tau,
                    device=device,
                )
            case _:
                raise ValueError(f"Unknown structure method: {structure_method!r}")

        # ── Episode state (initialised per reset) ──────────────────────────────
        self.gls: GraphLayoutState
        self.coords: np.ndarray  # float64 alias to self.gls.positions
        self.current_crossings: float
        self.current_structure: float
        self.current_potential: float
        self.initial_crossings: float
        self.initial_structure: float
        self.best_crossings: float
        self.steps: int = 0
        self.no_improve_steps: int = 0

    # ── Crossing / structure ───────────────────────────────────────────────────

    def _compute_crossings(self, coords: np.ndarray) -> float:
        return float(self.gls.compute_crossings()[0])

    def _compute_structure(self, coords: np.ndarray) -> float:
        return self.structure_loss(
            torch.tensor(coords, dtype=torch.float32, device=self.device)
        ).item()

    # ── Feature helpers ────────────────────────────────────────────────────────

    def _compute_edge_lengths(self, coords: np.ndarray) -> np.ndarray:
        if self.edge_index.shape[1] == 0:
            return np.array([], dtype=np.float32)
        diff = coords[self.edge_index[0]] - coords[self.edge_index[1]]
        return np.linalg.norm(diff, axis=1).astype(np.float32)

    def _get_node_features(self, coords: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [coords, self.degree.reshape(-1, 1)]
        ).astype(np.float32)

    def _get_edge_features(self, coords: np.ndarray) -> np.ndarray:
        lengths = self._compute_edge_lengths(coords)
        if len(lengths) == 0:
            return np.zeros((0, 1), dtype=np.float32)
        return lengths.reshape(-1, 1)

    def _get_initial_layout(self) -> np.ndarray:
        raw = get_initial_layout(
            self.initial_layout, self.num_nodes, self.neato_coords, self.graph
        )
        min_c = raw.min(axis=0)
        scale = (raw.max(axis=0) - min_c).max() or 1.0
        return (raw - min_c) / scale  # normalised to [0, 1]

    # ── Reset ──────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        initial_coords = self._get_initial_layout()
        self.gls = GraphLayoutState(
            self.graph, range(self.num_nodes), initial_coords
        )
        self.coords = self.gls.positions  # float64 alias; feature helpers cast as needed
        self.current_crossings = self._compute_crossings(self.coords)
        self.current_structure = self._compute_structure(self.coords)
        self.current_potential = -(
            self.current_crossings
            + self.structure_weight * self.current_structure
        )
        self.initial_crossings = self.current_crossings
        self.initial_structure = self.current_structure
        self.best_crossings = self.current_crossings
        self.steps = 0
        self.no_improve_steps = 0

        info = {
            "crossings":      self.current_crossings,
            "structure_loss": self.current_structure,
            "num_nodes":      self.num_nodes,
            "num_edges":      self.num_edges,
        }
        return self._get_node_features(self.coords), info

    # ── Policy interface ───────────────────────────────────────────────────────

    def get_graph_data(self):
        edge_features = self._get_edge_features(self.coords)
        return {
            "edge_index": torch.tensor(self.edge_index, dtype=torch.long),
            "edge_attr":  torch.tensor(edge_features, dtype=torch.float32),
            "num_nodes":  self.num_nodes,
            "degree":     torch.tensor(self.degree, dtype=torch.float32),
        }

    def get_coords(self) -> np.ndarray:
        return self.coords.copy()

    def get_policy_input(self, obs: np.ndarray, device) -> tuple:
        gd = self.get_graph_data()
        return (
            torch.tensor(obs, dtype=torch.float32, device=device),
            gd["edge_index"].to(device),
            gd["edge_attr"].to(device),
        )

    @staticmethod
    def make_batch_input(envs, obs_list, device):
        data_list = []
        for env, obs in zip(envs, obs_list):
            gd = env.get_graph_data()
            data_list.append(
                Data(
                    x=torch.tensor(obs, dtype=torch.float32, device=device),
                    edge_index=gd["edge_index"].to(device),
                    edge_attr=gd["edge_attr"].to(device),
                )
            )
        return Batch.from_data_list(data_list)

    @abstractmethod
    def step(self, action):
        ...
