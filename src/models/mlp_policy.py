"""
MLP-based Policy Network for Graph Layout Optimization
(Simple version without GNN for baseline comparison)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


class MLPPolicy(nn.Module):
    """
    Simple MLP policy that takes flattened coordinates as input.

    Note: This version requires fixed graph size. For variable-size graphs,
    use GNNPolicy instead.
    """

    def __init__(
        self,
        num_nodes: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        """
        Args:
            num_nodes: Number of nodes in the graph
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
        """
        super().__init__()

        self.num_nodes = num_nodes
        input_dim = num_nodes * 2  # Flattened coordinates

        # Shared feature extractor
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        self.shared = nn.Sequential(*layers)

        # Node selection head (actor)
        self.node_head = nn.Linear(hidden_dim, num_nodes)

        # Delta prediction head (actor)
        self.delta_mean = nn.Linear(hidden_dim, 2)
        self.delta_log_std = nn.Parameter(torch.zeros(2))

        # Value head (critic)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, coords: torch.Tensor):
        """
        Args:
            coords: Node coordinates [batch, num_nodes, 2]

        Returns:
            node_logits: [batch, num_nodes]
            delta_mean: [batch, 2]
            delta_log_std: [2]
            value: [batch, 1]
        """
        batch_size = coords.shape[0]
        x = coords.view(batch_size, -1)  # Flatten: [batch, num_nodes * 2]

        features = self.shared(x)

        node_logits = self.node_head(features)
        delta_mean = torch.tanh(self.delta_mean(features))  # Bound to [-1, 1]
        value = self.value_head(features)

        return node_logits, delta_mean, self.delta_log_std, value

    def get_action(self, coords: torch.Tensor, deterministic: bool = False):
        """
        Sample action from policy.

        Args:
            coords: Node coordinates [batch, num_nodes, 2] or [num_nodes, 2]
            deterministic: If True, return mean action

        Returns:
            action: Dict with "node" and "delta"
            log_prob: Log probability of action
            value: State value estimate
        """
        # Add batch dimension if needed
        if coords.dim() == 2:
            coords = coords.unsqueeze(0)

        node_logits, delta_mean, delta_log_std, value = self.forward(coords)

        # Node selection
        node_dist = Categorical(logits=node_logits)
        if deterministic:
            node = node_logits.argmax(dim=-1)
        else:
            node = node_dist.sample()

        # Delta selection
        delta_std = torch.exp(delta_log_std)
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
            "node": node.squeeze(0).item() if coords.shape[0] == 1 else node,
            "delta": delta.squeeze(0).cpu().numpy() if coords.shape[0] == 1 else delta,
        }

        return action, log_prob.squeeze(0), value.squeeze()

    def evaluate_action(self, coords: torch.Tensor, action_node: torch.Tensor, action_delta: torch.Tensor):
        """
        Evaluate given actions (for PPO update).

        Args:
            coords: [batch, num_nodes, 2]
            action_node: [batch]
            action_delta: [batch, 2]

        Returns:
            log_prob: [batch]
            entropy: [batch]
            value: [batch]
        """
        node_logits, delta_mean, delta_log_std, value = self.forward(coords)

        # Node distribution
        node_dist = Categorical(logits=node_logits)
        node_log_prob = node_dist.log_prob(action_node)
        node_entropy = node_dist.entropy()

        # Delta distribution
        delta_std = torch.exp(delta_log_std)
        delta_dist = Normal(delta_mean, delta_std)
        delta_log_prob = delta_dist.log_prob(action_delta).sum(dim=-1)
        delta_entropy = delta_dist.entropy().sum(dim=-1)

        log_prob = node_log_prob + delta_log_prob
        entropy = node_entropy + delta_entropy

        return log_prob, entropy, value.squeeze(-1)
