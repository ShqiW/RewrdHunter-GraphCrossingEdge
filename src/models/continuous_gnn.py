"""
GAT-based Policy Network for continuous action space.

Architecture:
1. GAT encodes node features (x, y, degree) with edge features (edge_length)
2. For each node, output 2D action mean (dx, dy)
3. Shared learnable log_std parameter
4. Value head: global mean pooling -> scalar

Action distribution: Independent Normal(mu, exp(log_std)) over all nodes x 2 dims
log_prob = sum over nodes and dims of Normal.log_prob (joint product of Gaussians)
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch_geometric.nn import GATConv, global_mean_pool

from src.models.base import BasePolicy
from src.tasks.base import BasePPOConfig

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


@dataclass
class ContinuousGNNConfig(BasePPOConfig):
    """Continuous GAT policy configuration"""
    num_gnn_layers: int = 3
    node_input_dim: int = 3   # x, y, degree
    edge_input_dim: int = 1   # edge_length
    num_heads: int = 4
    dropout: float = 0.1
    move_scale: float = 0.05  # clips sampled action to [-move_scale, move_scale]
    init_log_std: float = -1.0  # initial log std ~ std ≈ 0.37


class ContinuousGNNPolicy(BasePolicy):
    """
    Continuous action space GAT policy.

    Outputs a Gaussian distribution over per-node (dx, dy) deltas.
    All nodes are moved simultaneously each step.
    """

    def __init__(self, config: ContinuousGNNConfig, **kwargs):
        super().__init__()

        hidden_dim = config.hidden_dim
        num_heads = config.num_heads
        dropout = config.dropout

        self.move_scale = config.move_scale
        self.num_heads = num_heads

        # Node / edge projections
        self.node_proj = nn.Linear(config.node_input_dim, hidden_dim)
        self.edge_proj = nn.Linear(config.edge_input_dim, hidden_dim)

        # GAT layers
        self.gat_layers = nn.ModuleList([
            GATConv(
                hidden_dim,
                hidden_dim // num_heads,
                heads=num_heads,
                concat=True,
                edge_dim=hidden_dim,
                dropout=dropout,
            )
            for _ in range(config.num_gnn_layers)
        ])

        # Actor head: per-node 2D mean
        self.mu_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

        # Shared log_std (learnable scalar, clamped during forward)
        self.log_std = nn.Parameter(
            torch.full((2,), config.init_log_std))

        # Critic head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    # ── Shared encoder ─────────────────────────────────────────────────────────

    def encode(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
    ):
        """
        Args:
            node_features: [total_nodes, 3]
            edge_index:    [2, total_edges]
            edge_attr:     [total_edges, 1]
            batch:         [total_nodes] node-to-graph mapping

        Returns:
            node_embs:  [total_nodes, hidden_dim]
            graph_emb:  [num_graphs, hidden_dim]
        """
        x = self.node_proj(node_features)
        edge_emb = self.edge_proj(edge_attr) if edge_attr is not None and edge_attr.shape[0] > 0 else None

        for gat in self.gat_layers:
            x = F.elu(gat(x, edge_index, edge_attr=edge_emb))

        if batch is None:
            batch = torch.zeros(node_features.shape[0], dtype=torch.long,
                                device=node_features.device)
        graph_emb = global_mean_pool(x, batch)
        return x, graph_emb

    # ── BasePolicy interface ────────────────────────────────────────────────────

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
    ):
        """
        Returns:
            mu:    [total_nodes, 2]  — action mean per node
            std:   [2]               — shared std (broadcast over nodes)
            value: [num_graphs]      — critic estimate
        """
        node_embs, graph_emb = self.encode(node_features, edge_index, edge_attr, batch)
        mu = self.mu_head(node_embs)                           # [total_nodes, 2]
        log_std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()                                    # [2]
        value = self.value_head(graph_emb).squeeze(-1)        # [num_graphs]
        return mu, std, value

    def get_action(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        deterministic: bool = False,
    ):
        """
        Single-graph action sampling.

        Returns:
            action:   np.ndarray [num_nodes, 2] clipped to [-move_scale, move_scale]
            log_prob: scalar tensor — sum log prob over all nodes/dims
            value:    scalar tensor
        """
        mu, std, value = self.forward(node_features, edge_index, edge_attr)
        dist = Normal(mu, std.expand_as(mu))

        if deterministic:
            raw = mu
        else:
            raw = dist.rsample()

        # tanh squashing: action in (-move_scale, move_scale), no hard wall
        action = torch.tanh(raw) * self.move_scale
        # correct log_prob for tanh transform: log|da/du| = log(1 - tanh²) + log(scale)
        log_prob = (dist.log_prob(raw)
                    - torch.log(1 - action.pow(2) / self.move_scale ** 2 + 1e-6)
                    ).mean()

        return action.detach().cpu().numpy(), log_prob, value.squeeze()

    def get_action_batched(
        self,
        batch_input,
        deterministic: bool = False,
    ):
        """
        Batched action sampling from a PyG Batch.

        Returns:
            actions:   list of np.ndarray, each [num_nodes_i, 2]
            log_probs: [B] tensor
            values:    [B] tensor
        """
        node_embs, graph_embs = self.encode(
            batch_input.x,
            batch_input.edge_index,
            batch_input.edge_attr,
            batch=batch_input.batch,
        )
        mu_all = self.mu_head(node_embs)           # [total_nodes, 2]
        log_std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        values = self.value_head(graph_embs).squeeze(-1)  # [B]

        ptr = batch_input.ptr
        batch_size = batch_input.num_graphs
        actions, log_probs = [], []

        for i in range(batch_size):
            s, e = ptr[i].item(), ptr[i + 1].item()
            mu_i = mu_all[s:e]                             # [num_nodes_i, 2]
            dist = Normal(mu_i, std.expand_as(mu_i))
            raw = mu_i if deterministic else dist.rsample()
            action = torch.tanh(raw) * self.move_scale
            lp = (dist.log_prob(raw)
                  - torch.log(1 - action.pow(2) / self.move_scale ** 2 + 1e-6)
                  ).mean()
            actions.append(action.detach().cpu().numpy())
            log_probs.append(lp)

        return actions, torch.stack(log_probs), values

    def evaluate_action_batched(
        self,
        batched_data,
        actions: list[torch.Tensor],
    ):
        """
        Evaluate stored actions for PPO update.

        Args:
            batched_data: PyG Batch
            actions:      list of [num_nodes_i, 2] float tensors (one per graph)

        Returns:
            log_probs:  [B]
            entropies:  [B]  — sum entropy over all nodes/dims
            values:     [B]
        """
        node_embs, graph_embs = self.encode(
            batched_data.x,
            batched_data.edge_index,
            batched_data.edge_attr,
            batch=batched_data.batch,
        )
        mu_all = self.mu_head(node_embs)
        log_std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        values = self.value_head(graph_embs).squeeze(-1)

        ptr = batched_data.ptr
        batch_size = batched_data.num_graphs
        log_probs, entropies = [], []

        for i in range(batch_size):
            s, e = ptr[i].item(), ptr[i + 1].item()
            mu_i = mu_all[s:e]
            dist = Normal(mu_i, std.expand_as(mu_i))
            a_i = actions[i].to(mu_i.device)
            # a_i = tanh(raw) * move_scale  →  raw = atanh(a_i / move_scale)
            raw_i = torch.atanh((a_i / self.move_scale).clamp(-1 + 1e-6, 1 - 1e-6))
            lp = (dist.log_prob(raw_i)
                  - torch.log(1 - a_i.pow(2) / self.move_scale ** 2 + 1e-6)
                  ).mean()
            log_probs.append(lp)
            entropies.append(dist.entropy().mean())

        return torch.stack(log_probs), torch.stack(entropies), values
