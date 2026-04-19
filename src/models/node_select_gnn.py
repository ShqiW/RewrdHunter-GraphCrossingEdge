"""
Node-selecting GAT policy for iterative refinement.

At each step the policy:
  1. Scores every node with a learned head + crossing-count heuristic
  2. Samples (or argmax) one node to move
  3. Outputs (dx, dy) for that node only

Action returned: np.ndarray [n, 2] — non-zero only at selected node.
log_prob = log_prob(node_selection) + log_prob(delta | selected node)
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from src.models.continuous_gnn import ContinuousGNNConfig, LOG_STD_MIN, LOG_STD_MAX
from src.models.gat_refinement_policy import GATRefinementPolicy


@dataclass
class NodeSelectGNNConfig(ContinuousGNNConfig):
    """Config for node-selecting policy."""
    node_input_dim: int = 5          # x, y, crossing_count_norm, neighbor_flag, in_crossing
    crossing_score_alpha: float = 2.0  # weight for crossing-count heuristic in selection


class NodeSelectGNNPolicy(GATRefinementPolicy):
    """
    Extends GATRefinementPolicy with an explicit node-selection head.

    Selection logit[i] = learned_head(emb[i]) + alpha * crossing_count_norm[i]
    The crossing_count_norm is feature index 2 of the observation.

    Difference from Group 2's pure node_head:
      - Mixes learned score with explicit crossing-count signal (not purely learned)
      - Uses crossing-node context pooling for the delta prediction
    """

    def __init__(self, config: NodeSelectGNNConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.alpha = nn.Parameter(torch.tensor(float(config.crossing_score_alpha)))
        hidden_dim = config.hidden_dim

        # Node selection head: scores each node
        self.node_select_head = nn.Linear(hidden_dim, 1)

        # Crossing-context MLP: pools embeddings of crossing nodes, conditions delta
        self.crossing_ctx_proj = nn.Linear(hidden_dim, hidden_dim)
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _selection_logits(self, node_embs: torch.Tensor,
                          node_features: torch.Tensor) -> torch.Tensor:
        """
        Combine learned score + crossing-count heuristic.
        node_features[:, 2] = crossing_count_normalized
        """
        learned = self.node_select_head(node_embs).squeeze(-1)   # [n]
        crossing_score = node_features[:, 2]                      # [n], already normalised
        return learned + self.alpha * crossing_score              # [n]

    def _delta_for_node(self, node_embs: torch.Tensor,
                        node_features: torch.Tensor,
                        vt: int) -> torch.Tensor:
        """
        Predict (dx, dy) mean for node vt, conditioned on crossing-node context.
        """
        # Crossing-node context: mean of embeddings of in-crossing nodes
        in_crossing = node_features[:, 4] > 0.5          # [n] bool
        if in_crossing.any():
            ctx = self.crossing_ctx_proj(
                node_embs[in_crossing].mean(dim=0))       # [hidden]
        else:
            ctx = torch.zeros(node_embs.shape[1],
                              device=node_embs.device)

        combined = torch.cat([node_embs[vt], ctx], dim=-1)  # [hidden*2]
        return self.delta_head(combined)                      # [2]

    # ── single-env inference ──────────────────────────────────────────────────

    def get_action(self, node_features, edge_index, edge_attr=None,
                   deterministic=False):
        """
        Returns:
            action:   np.ndarray [n, 2]  — non-zero only at selected node
            log_prob: scalar tensor
            value:    scalar tensor
        """
        node_embs, graph_emb = self.encode(node_features, edge_index, edge_attr)
        values = self.value_head(graph_emb).squeeze(-1)

        # Node selection: always sample — argmax causes mode collapse (same node forever).
        # PPO gradient through log_p_node teaches which node to prefer.
        sel_logits = self._selection_logits(node_embs, node_features)  # [n]
        vt = int(Categorical(logits=sel_logits).sample().item())
        log_p_node = F.log_softmax(sel_logits, dim=0)[vt]

        # Delta: deterministic uses mean (no noise), stochastic samples
        mu_vt = self._delta_for_node(node_embs, node_features, vt)    # [2]
        log_std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        dist = Normal(mu_vt, std)
        raw = mu_vt if deterministic else dist.rsample()
        action_vt = torch.tanh(raw) * self.move_scale
        log_p_delta = (
            dist.log_prob(raw)
            - torch.log(1 - action_vt.pow(2) / self.move_scale ** 2 + 1e-6)
        ).sum()

        # Build full [n, 2] action (zeros elsewhere)
        n = node_features.shape[0]
        action_full = np.zeros((n, 2), dtype=np.float32)
        action_full[vt] = action_vt.detach().cpu().numpy()

        log_prob = log_p_node + log_p_delta
        return action_full, log_prob, values.squeeze()

    # ── batched inference ─────────────────────────────────────────────────────

    def get_action_batched(self, pyg_batch, deterministic=False):
        """
        Returns:
            actions:   list of np.ndarray [n_i, 2]
            log_probs: [B] tensor
            values:    [B] tensor
        """
        node_embs, graph_embs = self.encode(
            pyg_batch.x, pyg_batch.edge_index, pyg_batch.edge_attr,
            batch=pyg_batch.batch,
        )
        log_std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        values = self.value_head(graph_embs).squeeze(-1)

        ptr = pyg_batch.ptr
        B = pyg_batch.num_graphs
        actions, log_probs = [], []

        for i in range(B):
            s, e = ptr[i].item(), ptr[i + 1].item()
            embs_i = node_embs[s:e]          # [n_i, hidden]
            feats_i = pyg_batch.x[s:e]       # [n_i, 5]
            n_i = e - s

            # Node selection: always sample to avoid mode collapse.
            # PPO gradient through log_p_node teaches which node to prefer.
            sel_logits = self._selection_logits(embs_i, feats_i)  # [n_i]
            vt = int(Categorical(logits=sel_logits).sample().item())
            log_p_node = F.log_softmax(sel_logits, dim=0)[vt]

            # Delta
            mu_vt = self._delta_for_node(embs_i, feats_i, vt)
            dist = Normal(mu_vt, std)
            raw = mu_vt if deterministic else dist.rsample()
            action_vt = torch.tanh(raw) * self.move_scale
            log_p_delta = (
                dist.log_prob(raw)
                - torch.log(1 - action_vt.pow(2) / self.move_scale ** 2 + 1e-6)
            ).sum()

            action_full = np.zeros((n_i, 2), dtype=np.float32)
            action_full[vt] = action_vt.detach().cpu().numpy()
            actions.append(action_full)
            log_probs.append(log_p_node + log_p_delta)

        return actions, torch.stack(log_probs), values

    # ── PPO update ────────────────────────────────────────────────────────────

    def evaluate_action_batched(self, pyg_batch, actions_list):
        """
        Args:
            pyg_batch:    PyG Batch
            actions_list: list of [n_i, 2] tensors

        Returns:
            log_probs:  [B]
            entropies:  [B]
            values:     [B]
        """
        node_embs, graph_embs = self.encode(
            pyg_batch.x, pyg_batch.edge_index, pyg_batch.edge_attr,
            batch=pyg_batch.batch,
        )
        log_std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        values = self.value_head(graph_embs).squeeze(-1)

        ptr = pyg_batch.ptr
        B = pyg_batch.num_graphs
        log_probs, entropies = [], []

        for i in range(B):
            s, e = ptr[i].item(), ptr[i + 1].item()
            embs_i = node_embs[s:e]
            feats_i = pyg_batch.x[s:e]

            # Recover selected node from stored action (highest magnitude row)
            a_i = actions_list[i].to(embs_i.device)     # [n_i, 2]
            vt = int(a_i.norm(dim=-1).argmax().item())

            # Node selection log_prob
            sel_logits = self._selection_logits(embs_i, feats_i)
            log_p_node = F.log_softmax(sel_logits, dim=0)[vt]

            # Delta log_prob for selected node
            mu_vt = self._delta_for_node(embs_i, feats_i, vt)
            dist = Normal(mu_vt, std)
            a_vt = a_i[vt]
            raw_vt = torch.atanh((a_vt / self.move_scale).clamp(-1 + 1e-6, 1 - 1e-6))
            log_p_delta = (
                dist.log_prob(raw_vt)
                - torch.log(1 - a_vt.pow(2) / self.move_scale ** 2 + 1e-6)
            ).sum()

            log_probs.append(log_p_node + log_p_delta)
            cat_entropy = Categorical(logits=sel_logits).entropy()
            entropies.append(cat_entropy + dist.entropy().sum())

        return torch.stack(log_probs), torch.stack(entropies), values
