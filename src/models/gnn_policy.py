"""
GNN-based Policy Network for Graph Layout Optimization
(Handles variable-size graphs)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch


class GNNPolicy(nn.Module):
    """
    GNN-based policy that can handle variable-size graphs.

    Uses Graph Convolutional Network to encode node features,
    then outputs node selection probabilities and movement deltas.
    """

    def __init__(
        self,
        input_dim: int = 2,  # x, y coordinates
        hidden_dim: int = 128,
        num_gnn_layers: int = 3,
    ):
        """
        Args:
            input_dim: Input feature dimension (2 for coordinates)
            hidden_dim: Hidden layer dimension
            num_gnn_layers: Number of GNN layers
        """
        super().__init__()

        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_gnn_layers):
            self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim))

        # Node selection head (per-node scores)
        self.node_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Delta prediction head (conditioned on selected node)
        # Takes: node embedding + graph embedding
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
        self.delta_log_std = nn.Parameter(torch.zeros(2))

        # Value head (graph-level)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def encode(self, coords: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor = None):
        """
        Encode graph using GNN.

        Args:
            coords: Node coordinates [num_nodes, 2]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment [num_nodes] (for batched graphs)

        Returns:
            node_embeddings: [num_nodes, hidden_dim]
            graph_embedding: [batch_size, hidden_dim]
        """
        # Input projection
        x = self.input_proj(coords)

        # GNN message passing
        for gnn in self.gnn_layers:
            x = gnn(x, edge_index)
            x = F.relu(x)

        node_embeddings = x

        # Graph-level pooling
        if batch is None:
            batch = torch.zeros(coords.shape[0], dtype=torch.long, device=coords.device)
        graph_embedding = global_mean_pool(node_embeddings, batch)

        return node_embeddings, graph_embedding

    def forward(self, coords: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor = None):
        """
        Forward pass.

        Args:
            coords: [num_nodes, 2]
            edge_index: [2, num_edges]
            batch: [num_nodes] batch assignment

        Returns:
            node_logits: [num_nodes] (logits for node selection)
            graph_embedding: [batch_size, hidden_dim]
            node_embeddings: [num_nodes, hidden_dim]
            value: [batch_size, 1]
        """
        node_embeddings, graph_embedding = self.encode(coords, edge_index, batch)

        # Node selection scores
        node_logits = self.node_head(node_embeddings).squeeze(-1)

        # Value estimate
        value = self.value_head(graph_embedding)

        return node_logits, graph_embedding, node_embeddings, value

    def get_action(
        self,
        coords: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor = None,
        deterministic: bool = False
    ):
        """
        Sample action from policy.

        Args:
            coords: [num_nodes, 2]
            edge_index: [2, num_edges]
            batch: [num_nodes] batch assignment (None for single graph)
            deterministic: If True, return argmax action

        Returns:
            action: Dict with "node" and "delta"
            log_prob: Log probability of action
            value: State value estimate
        """
        node_logits, graph_embedding, node_embeddings, value = self.forward(
            coords, edge_index, batch
        )

        # For single graph (no batching)
        if batch is None:
            batch = torch.zeros(coords.shape[0], dtype=torch.long, device=coords.device)

        # Node selection
        node_dist = Categorical(logits=node_logits)
        if deterministic:
            node = node_logits.argmax()
        else:
            node = node_dist.sample()

        # Get selected node embedding
        selected_node_emb = node_embeddings[node]

        # Expand graph embedding to match (for single graph case)
        batch_idx = batch[node]
        selected_graph_emb = graph_embedding[batch_idx]

        # Concatenate node and graph embeddings for delta prediction
        combined = torch.cat([selected_node_emb, selected_graph_emb], dim=-1)

        # Delta prediction
        delta_mean = torch.tanh(self.delta_head(combined))
        delta_std = torch.exp(self.delta_log_std)
        delta_dist = Normal(delta_mean, delta_std)

        if deterministic:
            delta = delta_mean
        else:
            delta = delta_dist.sample()
        delta = torch.clamp(delta, -1.0, 1.0)

        # Compute log probabilities
        node_log_prob = node_dist.log_prob(node)
        delta_log_prob = delta_dist.log_prob(delta).sum(dim=-1)
        log_prob = node_log_prob + delta_log_prob

        action = {
            "node": node.item(),
            "delta": delta.cpu().numpy(),
        }

        return action, log_prob, value.squeeze()

    def evaluate_action(
        self,
        coords: torch.Tensor,
        edge_index: torch.Tensor,
        action_node: torch.Tensor,
        action_delta: torch.Tensor,
        batch: torch.Tensor = None,
        ptr: torch.Tensor = None,
    ):
        """
        Evaluate given actions (for PPO update).

        Args:
            coords: [total_nodes, 2] (batched)
            edge_index: [2, total_edges] (batched)
            action_node: [batch_size] node indices (local to each graph)
            action_delta: [batch_size, 2]
            batch: [total_nodes] batch assignment
            ptr: [batch_size + 1] pointers to graph boundaries

        Returns:
            log_prob: [batch_size]
            entropy: [batch_size]
            value: [batch_size]
        """
        node_logits, graph_embedding, node_embeddings, value = self.forward(
            coords, edge_index, batch
        )

        batch_size = graph_embedding.shape[0]

        # We need to handle the fact that action_node is relative to each graph
        # Convert to global indices
        if ptr is not None:
            global_node_indices = action_node + ptr[:-1]
        else:
            global_node_indices = action_node

        # Compute node log probs per graph
        # This is tricky with variable-size graphs; we'll iterate for now
        node_log_probs = []
        node_entropies = []

        for i in range(batch_size):
            mask = (batch == i)
            graph_node_logits = node_logits[mask]
            dist = Categorical(logits=graph_node_logits)
            local_action = action_node[i]
            node_log_probs.append(dist.log_prob(local_action))
            node_entropies.append(dist.entropy())

        node_log_prob = torch.stack(node_log_probs)
        node_entropy = torch.stack(node_entropies)

        # Delta log prob
        selected_node_emb = node_embeddings[global_node_indices]
        combined = torch.cat([selected_node_emb, graph_embedding], dim=-1)
        delta_mean = torch.tanh(self.delta_head(combined))
        delta_std = torch.exp(self.delta_log_std)
        delta_dist = Normal(delta_mean, delta_std)

        delta_log_prob = delta_dist.log_prob(action_delta).sum(dim=-1)
        delta_entropy = delta_dist.entropy().sum(dim=-1)

        log_prob = node_log_prob + delta_log_prob
        entropy = node_entropy + delta_entropy

        return log_prob, entropy, value.squeeze(-1)
