"""
GAT policy for iterative refinement.

Same GAT encoder as ContinuousGNNPolicy, but outputs (dx, dy) only for the
current node vt (identified by obs[:, 2] == 1) instead of all nodes at once.
"""
import numpy as np
import torch
from torch.distributions import Normal

from src.models.continuous_gnn import ContinuousGNNPolicy, LOG_STD_MIN, LOG_STD_MAX


class GATRefinementPolicy(ContinuousGNNPolicy):
    """
    Actor-critic for single-node refinement.

    Reuses the GAT encoder from ContinuousGNNPolicy.  At each step the env
    marks exactly one node as vt (obs[:, 2] == 1); this policy extracts that
    node's embedding and outputs a 2-D (dx, dy) action.

    Input node features: [x, y, is_vt, is_neighbor, in_crossing]  (dim=5)
    Action: np.ndarray [2], scaled to (-move_scale, move_scale) via tanh
    """

    # ── single-env inference ──────────────────────────────────────────────────

    def get_action(self, node_features, edge_index, edge_attr=None,
                   deterministic=False):
        """
        Args:
            node_features: [N, 5] tensor
            edge_index:    [2, E] tensor
            edge_attr:     [E, 1] tensor or None
        Returns:
            action:   np.ndarray [2]
            log_prob: scalar tensor
            value:    scalar tensor
        """
        mu, std, value = self.forward(node_features, edge_index, edge_attr)
        vt_idx = (node_features[:, 2] > 0.5).nonzero(as_tuple=True)[0][0]
        mu_vt = mu[vt_idx]  # [2]
        dist = Normal(mu_vt, std)
        raw = mu_vt if deterministic else dist.rsample()
        action = torch.tanh(raw) * self.move_scale
        log_prob = (
            dist.log_prob(raw)
            - torch.log(1 - action.pow(2) / self.move_scale ** 2 + 1e-6)
        ).sum()
        return action.detach().cpu().numpy(), log_prob, value.squeeze()

    # ── batched inference (called from rollout buffer) ────────────────────────

    def get_action_batched(self, pyg_batch, deterministic=False):
        """
        Args:
            pyg_batch: PyG Batch with .x [total_N, 5], .edge_index, .edge_attr, .batch
        Returns:
            actions:   list of np.ndarray [2], one per graph
            log_probs: [B] tensor
            values:    [B] tensor
        """
        node_embs, graph_embs = self.encode(
            pyg_batch.x, pyg_batch.edge_index, pyg_batch.edge_attr,
            batch=pyg_batch.batch,
        )
        mu_all = self.mu_head(node_embs)       # [total_N, 2]
        log_std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        values = self.value_head(graph_embs).squeeze(-1)  # [B]

        ptr = pyg_batch.ptr
        B = pyg_batch.num_graphs
        actions, log_probs = [], []

        for i in range(B):
            s, e = ptr[i].item(), ptr[i + 1].item()
            vt_mask = pyg_batch.x[s:e, 2] > 0.5
            vt_local = vt_mask.nonzero(as_tuple=True)[0][0]
            mu_vt = mu_all[s + vt_local]  # [2]
            dist = Normal(mu_vt, std)
            raw = mu_vt if deterministic else dist.rsample()
            action = torch.tanh(raw) * self.move_scale
            lp = (
                dist.log_prob(raw)
                - torch.log(1 - action.pow(2) / self.move_scale ** 2 + 1e-6)
            ).sum()
            actions.append(action.detach().cpu().numpy())
            log_probs.append(lp)

        return actions, torch.stack(log_probs), values

    # ── PPO update ────────────────────────────────────────────────────────────

    def evaluate_action_batched(self, pyg_batch, actions_tensor):
        """
        Args:
            pyg_batch:      PyG Batch
            actions_tensor: [B, 2] stored actions (one per graph's vt node)
        Returns:
            log_probs:  [B]
            entropies:  [B]
            values:     [B]
        """
        node_embs, graph_embs = self.encode(
            pyg_batch.x, pyg_batch.edge_index, pyg_batch.edge_attr,
            batch=pyg_batch.batch,
        )
        mu_all = self.mu_head(node_embs)
        log_std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        values = self.value_head(graph_embs).squeeze(-1)

        ptr = pyg_batch.ptr
        B = pyg_batch.num_graphs
        log_probs, entropies = [], []

        for i in range(B):
            s, e = ptr[i].item(), ptr[i + 1].item()
            vt_mask = pyg_batch.x[s:e, 2] > 0.5
            vt_local = vt_mask.nonzero(as_tuple=True)[0][0]
            mu_vt = mu_all[s + vt_local]  # [2]
            dist = Normal(mu_vt, std)
            a_i = actions_tensor[i].to(mu_vt.device)
            raw_i = torch.atanh((a_i / self.move_scale).clamp(-1 + 1e-6, 1 - 1e-6))
            lp = (
                dist.log_prob(raw_i)
                - torch.log(1 - a_i.pow(2) / self.move_scale ** 2 + 1e-6)
            ).sum()
            log_probs.append(lp)
            entropies.append(dist.entropy().sum())

        return torch.stack(log_probs), torch.stack(entropies), values
