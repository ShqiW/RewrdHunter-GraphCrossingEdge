"""
Transformer-based policy for sequential node placement.

Architecture (V2):
    1. Global summary:   mean-pool projection of ALL placed nodes  — O(n), no attention
    2. Neighbor encoder: Transformer over v_t's placed neighbors   — O(deg²), deg ≈ 2-5
    3. Combiner:         Linear(2d → d) + ReLU
    4. Actor head:       context → μ (Sigmoid), log_σ (clamped)
    5. Critic head:      context → scalar value

V1 ran Transformer over all placed nodes (O(n²) attention), which became the
dominant cost as more nodes were placed.  V2 fixes this by restricting attention
to the small neighbor subsequence while retaining global layout awareness via
the cheap mean-pool summary.
"""
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
from torch.distributions import Normal

from src.tasks.base import BaseModelConfig


@dataclass
class TransformerConfig(BaseModelConfig):
    """Transformer placement policy configuration"""
    type: str = "transformer"
    hidden_dim: int = 128
    num_heads: int = 4
    num_encoder_layers: int = 3
    dropout: float = 0.1
    log_sigma_min: float = -4.0
    log_sigma_max: float = 0.0  # sigma ∈ (exp(-4), 1) ≈ (0.018, 1.0)


class TransformerPlacementPolicy(nn.Module):
    """
    Actor-critic policy for sequential node placement.

    Input (per step):
        node_features: [num_placed, 3] — (x, y, is_neighbor_of_vt)

    Output:
        action: [2] — (x, y) sampled from Normal(μ, σ), clipped to [0,1]
        log_prob: scalar
        value: scalar
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()

        d = config.hidden_dim
        nhead = config.num_heads
        nlayers = config.num_encoder_layers
        dropout = config.dropout
        self.log_sigma_min = config.log_sigma_min
        self.log_sigma_max = config.log_sigma_max

        # Shared node feature projection (used by both paths)
        self.input_proj = nn.Linear(3, d)

        # Neighbor-only Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=nhead,
            dim_feedforward=d * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer,
                                             num_layers=nlayers,
                                             enable_nested_tensor=False)

        # Null token: used when there are no placed nodes or no placed neighbors
        self.null_token = nn.Parameter(torch.zeros(d))

        # Combiner: merge neighbor context + global summary → single context
        self.combiner = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.ReLU(),
        )

        # Actor: context → μ (via Sigmoid) and log_σ
        self.actor_mu = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 2),
            nn.Sigmoid(),
        )
        self.actor_log_sigma = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 2),
        )

        # Critic: context → scalar value
        self.critic = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 1),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_and_pool(self,
                         batch_feats: List[torch.Tensor]) -> torch.Tensor:
        """
        Pad a batch of variable-size feature tensors, run through the Transformer
        encoder, and return mean-pooled embeddings.

        Args:
            batch_feats: list of B tensors, each [ni, 3]  (ni may be 0)

        Returns:
            ctx: [B, d] — mean-pooled encoder output; null_token for empty sequences
        """
        B = len(batch_feats)
        device = self.null_token.device if B == 0 else batch_feats[0].device
        d = self.null_token.shape[0]

        sizes = [f.shape[0] for f in batch_feats]
        max_n = max(sizes) if sizes and max(sizes) > 0 else 0

        if max_n == 0:
            # All sequences empty — return null token for every item
            return self.null_token.unsqueeze(0).expand(B, d).clone()

        padded = torch.zeros(B, max_n, 3, device=device)
        mask = torch.ones(B, max_n, dtype=torch.bool,
                          device=device)  # True=padding
        for i, (feats, sz) in enumerate(zip(batch_feats, sizes)):
            if sz > 0:
                padded[i, :sz] = feats
                mask[i, :sz] = False

        x = self.input_proj(padded)  # [B, max_n, d]

        # Empty sequences crash the Transformer — replace with null token
        all_empty = mask.all(dim=1)  # [B]
        if all_empty.any():
            x = x.clone()
            mask = mask.clone()
            x[all_empty,
              0] = self.null_token.unsqueeze(0).expand(int(all_empty.sum()),
                                                       -1)
            mask[all_empty, 0] = False

        embs = self.encoder(x, src_key_padding_mask=mask)  # [B, max_n, d]

        # Mean pool valid positions
        valid = ~mask  # [B, max_n]
        w = valid.float().unsqueeze(-1)  # [B, max_n, 1]
        ctx = (embs * w).sum(1) / w.sum(1).clamp(min=1)  # [B, d]
        if all_empty.any():
            ctx = ctx.clone()
            ctx[all_empty] = self.null_token
        return ctx

    def _compute_context(self,
                         batch_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute the combined context vector for a batch of states.

        Two-path approach:
          - Neighbor path:  Transformer over placed graph-neighbors of v_t  (O(deg²))
          - Global path:    mean-pool projection of all placed nodes         (O(n))

        Args:
            batch_features: list of B tensors, each [ni, 3]

        Returns:
            context: [B, d]
        """
        B = len(batch_features)
        device = self.null_token.device if B == 0 else batch_features[0].device
        d = self.null_token.shape[0]

        # ── Global summary ────────────────────────────────────────────────────
        # Project all placed nodes and mean-pool (no attention, O(n))
        global_ctx = torch.zeros(B, d, device=device)
        for i, feats in enumerate(batch_features):
            if feats.shape[0] > 0:
                global_ctx[i] = self.input_proj(feats).mean(dim=0)
            else:
                global_ctx[i] = self.null_token

        # ── Neighbor Transformer ──────────────────────────────────────────────
        # Extract only placed neighbors of v_t per batch item
        nb_feats: List[torch.Tensor] = []
        for feats in batch_features:
            if feats.shape[0] > 0:
                nb_mask = feats[:, 2] > 0.5
                nb_feats.append(feats[nb_mask] if nb_mask.any() else torch.
                                zeros(0, 3, device=device))
            else:
                nb_feats.append(torch.zeros(0, 3, device=device))

        nb_ctx = self._encode_and_pool(nb_feats)  # [B, d]

        # ── Combine ────────────────────────────────────────────────────────────
        return self.combiner(torch.cat([nb_ctx, global_ctx], dim=-1))  # [B, d]

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def forward(self, batch_features: List[torch.Tensor]):
        """
        Forward pass for a batch of states.

        Args:
            batch_features: list of B tensors, each [ni, 3]

        Returns:
            mu:    [B, 2]
            sigma: [B, 2]
            value: [B]
        """
        context = self._compute_context(batch_features)
        mu = self.actor_mu(context)
        log_sigma = self.actor_log_sigma(context).clamp(
            self.log_sigma_min, self.log_sigma_max)
        sigma = log_sigma.exp()
        value = self.critic(context).squeeze(-1)
        return mu, sigma, value

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def get_action(self,
                   node_features: torch.Tensor,
                   deterministic: bool = False):
        with torch.no_grad():
            mu, sigma, value = self.forward([node_features])
            mu, sigma, value = mu[0], sigma[0], value[0]
            if deterministic:
                action_t = mu
                log_prob = Normal(mu, sigma).log_prob(action_t).sum()
            else:
                dist = Normal(mu, sigma)
                action_t = dist.sample()
                log_prob = dist.log_prob(action_t).sum()
        return action_t.clamp(0.0, 1.0).cpu().numpy(), log_prob, value

    def get_action_batched(self,
                           batch_obs: List[torch.Tensor],
                           deterministic: bool = False):
        mu, sigma, value = self.forward(batch_obs)
        if deterministic:
            actions_t = mu
            log_probs = Normal(mu, sigma).log_prob(actions_t).sum(dim=-1)
        else:
            dist = Normal(mu, sigma)
            actions_t = dist.sample()
            log_probs = dist.log_prob(actions_t).sum(dim=-1)
        actions_t = actions_t.clamp(0.0, 1.0)
        actions = [actions_t[i].cpu().numpy() for i in range(len(batch_obs))]
        return actions, log_probs, value

    # ------------------------------------------------------------------
    # PPO evaluation — fast path with pre-padded tensors
    # ------------------------------------------------------------------

    def evaluate_action_prepadded(
        self,
        padded: torch.Tensor,
        mask: torch.Tensor,
        batch_actions: torch.Tensor,
    ):
        """
        Evaluate log_prob and entropy using pre-padded tensors.

        Args:
            padded:        [B, max_n, 3]
            mask:          [B, max_n] — True = padding
            batch_actions: [B, 2]

        Returns:
            log_probs: [B], entropies: [B], values: [B]
        """
        B, max_n, _ = padded.shape
        device = padded.device
        d = self.null_token.shape[0]

        valid = ~mask  # [B, max_n]

        # ── Single projection for both paths ──────────────────────────────────
        x_all = self.input_proj(padded)   # [B, max_n, d]  (called only once)

        # ── Global summary (vectorised) ────────────────────────────────────────
        w_all      = valid.float().unsqueeze(-1)                       # [B, max_n, 1]
        global_ctx = (x_all * w_all).sum(1) / w_all.sum(1).clamp(min=1)  # [B, d]
        no_nodes   = (~valid).all(1)
        if no_nodes.any():
            global_ctx = global_ctx.clone()
            global_ctx[no_nodes] = self.null_token

        # ── Neighbor Transformer (fully vectorised, no Python loop) ───────────
        nb_flag   = valid & (padded[:, :, 2] > 0.5)   # [B, max_n]
        nb_counts = nb_flag.sum(1)                     # [B]
        max_deg   = int(nb_counts.max().item()) if nb_counts.max() > 0 else 0

        if max_deg == 0:
            nb_ctx = self.null_token.unsqueeze(0).expand(B, d).clone()
        else:
            # Sort each row so neighbor positions come first, then gather
            # projected embeddings — no Python loop, no second input_proj call.
            sort_idx = nb_flag.long().argsort(dim=1, descending=True)   # [B, max_n]
            nb_x = torch.gather(
                x_all, 1,
                sort_idx[:, :max_deg].unsqueeze(-1).expand(-1, -1, d)
            )  # [B, max_deg, d]

            # Mask: positions ≥ nb_counts[i] are padding
            row_idx = torch.arange(max_deg, device=device).unsqueeze(0)  # [1, max_deg]
            nb_mask = row_idx >= nb_counts.unsqueeze(1)                   # [B, max_deg]

            all_empty = nb_mask.all(1)
            if all_empty.any():
                nb_x    = nb_x.clone()
                nb_mask = nb_mask.clone()
                nb_x[all_empty, 0]    = self.null_token.unsqueeze(0).expand(int(all_empty.sum()), -1)
                nb_mask[all_empty, 0] = False

            nb_embs = self.encoder(nb_x, src_key_padding_mask=nb_mask)  # [B, max_deg, d]
            w_nb    = (~nb_mask).float().unsqueeze(-1)
            nb_ctx  = (nb_embs * w_nb).sum(1) / w_nb.sum(1).clamp(min=1)
            if all_empty.any():
                nb_ctx = nb_ctx.clone()
                nb_ctx[all_empty] = self.null_token

        # ── Combine & heads ───────────────────────────────────────────────────
        context = self.combiner(torch.cat([nb_ctx, global_ctx], dim=-1))
        mu = self.actor_mu(context)
        log_sigma = self.actor_log_sigma(context).clamp(
            self.log_sigma_min, self.log_sigma_max)
        sigma = log_sigma.exp()
        value = self.critic(context).squeeze(-1)

        dist = Normal(mu, sigma)
        log_probs = dist.log_prob(batch_actions).sum(-1)
        entropies = dist.entropy().sum(-1)
        return log_probs, entropies, value

    # ------------------------------------------------------------------
    # PPO evaluation — list-based interface (calls forward)
    # ------------------------------------------------------------------

    def evaluate_action_batched(
        self,
        batch_features: List[torch.Tensor],
        batch_actions: torch.Tensor,
    ):
        mu, sigma, value = self.forward(batch_features)
        dist = Normal(mu, sigma)
        log_probs = dist.log_prob(batch_actions).sum(dim=-1)
        entropies = dist.entropy().sum(dim=-1)
        return log_probs, entropies, value
