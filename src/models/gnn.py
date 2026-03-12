"""
GAT-based Policy Network for discrete action space.

Architecture:
1. GAT encodes node features (x, y, degree) with edge features (edge_length, is_crossing)
2. For each node, output 8 direction logits
3. Flatten to num_nodes * 8 action distribution
4. Value head outputs state value
"""
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.nn import GATConv, global_mean_pool

from src.tasks.base import BaseModelConfig


NUM_DIRECTIONS = 8


@dataclass
class GNNConfig(BaseModelConfig):
    """GAT model configuration"""
    type: str = "gnn"
    hidden_dim: int = 128
    num_gnn_layers: int = 3
    node_input_dim: int = 3  # x, y, degree
    edge_input_dim: int = 1  # edge_length
    num_heads: int = 4
    dropout: float = 0.1


class DiscreteGNNPolicy(nn.Module):
    """
    Discrete action space GAT policy.

    State Space (from slides):
    - Node features: [x, y, degree]
    - Edge features: [edge_length]

    Output: Single Categorical distribution over num_nodes * 8 actions
    """

    def __init__(
        self,
        config: GNNConfig = None,
        node_input_dim: int = 3,
        edge_input_dim: int = 2,
        hidden_dim: int = 128,
        num_gnn_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()

        if config is not None:
            node_input_dim = config.node_input_dim
            edge_input_dim = config.edge_input_dim
            hidden_dim = config.hidden_dim
            num_gnn_layers = config.num_gnn_layers
            num_heads = config.num_heads
            dropout = config.dropout

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Node feature projection
        self.node_proj = nn.Linear(node_input_dim, hidden_dim)

        # Edge feature projection
        self.edge_proj = nn.Linear(edge_input_dim, hidden_dim)

        # GAT layers with edge features
        self.gat_layers = nn.ModuleList()
        for i in range(num_gnn_layers):
            # GAT with edge_dim for edge features
            self.gat_layers.append(
                GATConv(
                    hidden_dim,
                    hidden_dim // num_heads,
                    heads=num_heads,
                    concat=True,
                    edge_dim=hidden_dim,
                    dropout=dropout,
                )
            )

        # Action head: each node outputs 8 direction logits
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, NUM_DIRECTIONS),
        )

        # Value head: graph-level value estimate
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor = None,
        batch: torch.Tensor = None,
    ):
        """
        GAT encoding with edge features.

        Args:
            node_features: [num_nodes, 3] - (x, y, degree)
            edge_index: [2, num_edges]
            edge_attr: [num_edges, 2] - (edge_length, is_crossing)
            batch: [num_nodes] - node-to-graph mapping for batched input

        Returns:
            node_embs: [num_nodes, hidden_dim]
            graph_emb: [num_graphs, hidden_dim]
        """
        # Project node features
        x = self.node_proj(node_features)

        # Project edge features
        if edge_attr is not None and edge_attr.shape[0] > 0:
            edge_emb = self.edge_proj(edge_attr)
        else:
            edge_emb = None

        # GAT layers
        for gat in self.gat_layers:
            x = gat(x, edge_index, edge_attr=edge_emb)
            x = F.elu(x)

        node_embs = x

        # Graph-level pooling
        if batch is None:
            # Single graph: all nodes belong to graph 0
            batch = torch.zeros(node_features.shape[0], dtype=torch.long, device=node_features.device)
        graph_emb = global_mean_pool(node_embs, batch)

        return node_embs, graph_emb

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor = None,
    ):
        """
        Forward pass.

        Args:
            node_features: [num_nodes, 3]
            edge_index: [2, num_edges]
            edge_attr: [num_edges, 2]

        Returns:
            action_logits: [num_nodes * 8]
            value: [1]
            node_embs: [num_nodes, hidden_dim]
        """
        node_embs, graph_emb = self.encode(node_features, edge_index, edge_attr)

        # Per-node 8-direction logits: [num_nodes, 8]
        node_action_logits = self.action_head(node_embs)

        # Flatten to [num_nodes * 8]
        action_logits = node_action_logits.view(-1)

        # Graph-level value
        value = self.value_head(graph_emb)

        return action_logits, value, node_embs

    def get_action(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor = None,
        deterministic: bool = False,
    ):
        """
        Sample action.

        Args:
            node_features: [num_nodes, 3] - (x, y, degree)
            edge_index: [2, num_edges]
            edge_attr: [num_edges, 2] - (edge_length, is_crossing)
            deterministic: whether to use argmax

        Returns:
            action: int (node_id * 8 + direction_id)
            log_prob: action log probability
            value: state value
        """
        action_logits, value, _ = self.forward(node_features, edge_index, edge_attr)

        dist = Categorical(logits=action_logits)

        if deterministic:
            action = action_logits.argmax()
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)

        return action.item(), log_prob, value.squeeze()

    def evaluate_action(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        action: torch.Tensor,
        edge_attr: torch.Tensor = None,
    ):
        """
        Evaluate given action (single graph).

        Args:
            node_features: [num_nodes, 3]
            edge_index: [2, num_edges]
            action: int tensor
            edge_attr: [num_edges, 2]

        Returns:
            log_prob: action log probability
            entropy: distribution entropy
            value: state value
        """
        action_logits, value, _ = self.forward(node_features, edge_index, edge_attr)

        dist = Categorical(logits=action_logits)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return log_prob, entropy, value.squeeze()

    def evaluate_action_batched(
        self,
        batched_data,
        actions: torch.Tensor,
    ):
        """
        Evaluate actions for a batch of graphs.

        Args:
            batched_data: PyG Batch object containing:
                - x: [total_nodes, 3] batched node features
                - edge_index: [2, total_edges] batched edge indices
                - edge_attr: [total_edges, 2] batched edge features
                - batch: [total_nodes] node-to-graph mapping
                - ptr: [batch_size + 1] graph boundaries
            actions: [batch_size] action for each graph

        Returns:
            log_probs: [batch_size] log probability for each action
            entropies: [batch_size] entropy for each graph
            values: [batch_size] value for each graph
        """
        # Encode all graphs in one forward pass
        node_embs, graph_embs = self.encode(
            batched_data.x,
            batched_data.edge_index,
            batched_data.edge_attr,
            batch=batched_data.batch,
        )

        # Per-node action logits: [total_nodes, 8]
        node_action_logits = self.action_head(node_embs)

        # Graph-level values: [batch_size, 1]
        values = self.value_head(graph_embs).squeeze(-1)

        # Split action logits per graph and compute log_probs/entropies
        batch_size = batched_data.num_graphs
        ptr = batched_data.ptr  # [batch_size + 1]

        log_probs = []
        entropies = []

        for i in range(batch_size):
            # Get this graph's node action logits
            start, end = ptr[i].item(), ptr[i + 1].item()
            graph_node_logits = node_action_logits[start:end]  # [num_nodes_i, 8]

            # Flatten to [num_nodes_i * 8]
            action_logits = graph_node_logits.view(-1)

            # Create distribution and evaluate
            dist = Categorical(logits=action_logits)
            log_probs.append(dist.log_prob(actions[i]))
            entropies.append(dist.entropy())

        log_probs = torch.stack(log_probs)
        entropies = torch.stack(entropies)

        return log_probs, entropies, values
