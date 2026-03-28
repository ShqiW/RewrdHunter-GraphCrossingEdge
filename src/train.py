"""
Unified training module.

Shared PPO logic (GAE, loss, optimizer, checkpointing) lives here.
Environment-specific rollout collection and batch evaluation are
delegated to the respective RolloutBuffer subclass.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import asdict
from pathlib import Path
from tqdm import tqdm

from src.tasks.base import BaseArgs
from src.data.rome import RomeDataset
from src.data.BaseRolloutBuffer import BaseRolloutBuffer
from src.models.base import BasePolicy
from src.envs.base import BaseGraphEnv
from investigation.logger.logger import DynamicCSVLogger

# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────


def _compute_gae(rewards, values, dones, next_value, gamma, gae_lambda,
                 device):
    """Generalised Advantage Estimation (shared by all buffer types)."""
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        next_val = next_value if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
    returns = advantages + torch.tensor(
        values, dtype=torch.float32, device=device)
    return advantages, returns


def _save_checkpoint(save_path, model, optimizer, update, args, final=False):
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / ("final_model.pt"
                       if final else f"checkpoint_{update:07d}.pt")
    torch.save(
        {
            "policy_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": update,
            "args": asdict(args),
        }, path)
    print(f"  Saved to {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Unified train() entry point
# ──────────────────────────────────────────────────────────────────────────────


def train(
    args: BaseArgs,
    device: torch.device,
    model: BasePolicy,
    dataset: RomeDataset,
    env: BaseGraphEnv,
    buffer: BaseRolloutBuffer,
):
    """
    Train model with PPO.

    Environment-specific differences (rollout collection, batch evaluation)
    are encapsulated in the buffer.  Everything else is shared.
    """

    optimizer = optim.Adam(model.parameters(), lr=args.model.lr)
    logger = DynamicCSVLogger("history.csv")
    start_update = 0

    compute_gae_fn = lambda rewards, values, dones, next_value: _compute_gae(
        rewards, values, dones, next_value, args.model.gamma, args.model.
        gae_lambda, device)

    # ── Resume from checkpoint ────────────────────────────────────────────────
    save_path = Path(args.save_path)
    ckpt_path = save_path / "final_model.pt"
    if not ckpt_path.exists():
        checkpoints = sorted(save_path.glob("checkpoint_*.pt"),
                             key=lambda x: int(x.stem.split("_")[-1]))
        if checkpoints:
            ckpt_path = checkpoints[-1]
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["policy_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_update = ckpt["global_step"]
        logger.resync(start_update, "global_step")
        print(f"Loaded checkpoint from {ckpt_path}")
        print(f"Resuming from update {start_update}")

    n_updates = args.total_timesteps // args.model.n_steps
    print(f"Starting PPO training: {args.name}")
    print(f"Total timesteps: {args.total_timesteps}, Updates: {n_updates}")
    if start_update > 0:
        print(f"Resuming from update: {start_update}")
    print(f"Device: {device}")
    print("=" * 60)

    for update in tqdm(
            range(start_update + 1, n_updates + 1),
            desc="Training",
            initial=start_update,
            total=n_updates,
    ):
        # ── Collect rollouts + compute GAE ───────────────────────────────────
        advantages, returns, collect_log = buffer.collect(
            model,
            env,
            args.model.n_steps,
            args.model.n_envs,
            device,
            dataset,
            args,
            compute_gae_fn,
        )

        # ── Normalise advantages ─────────────────────────────────────────────
        adv_std = advantages.std()
        if adv_std > 1e-6:
            advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)
        ret_mean = returns.mean()
        ret_std = returns.std() + 1e-8

        # ── Prepare batch data (sequential: pre-pad; discrete: no-op) ────────
        buffer.prepare_update()

        total_loss = total_pg = total_vl = total_ent = 0
        indices = np.arange(len(buffer))

        # ── PPO epochs ───────────────────────────────────────────────────────
        for _ in range(args.model.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, len(buffer), args.model.batch_size):
                end = min(start + args.model.batch_size, len(buffer))
                bi = indices[start:end]

                log_probs, entropies, values = model.evaluate_action_batched(
                    *buffer.get_batch_data(bi))
                old_lp = buffer.get_old_log_probs(bi)
                batch_adv = advantages[bi]
                batch_ret_norm = (returns[bi] - ret_mean) / ret_std

                ratio = torch.exp(log_probs - old_lp)
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1 - args.model.clip_eps,
                                    1 + args.model.clip_eps) * batch_adv
                pg_loss = -torch.min(surr1, surr2).mean()
                vl = nn.functional.mse_loss(values, batch_ret_norm)
                ent_loss = -entropies.mean()
                loss = pg_loss + args.model.value_coef * vl + args.model.entropy_coef * ent_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),
                                         args.model.max_grad_norm)
                optimizer.step()

                total_loss += loss.item()
                total_pg += pg_loss.item()
                total_vl += vl.item()
                total_ent += (-ent_loss.item())

        n_batch_updates = args.model.n_epochs * (
            len(buffer) // args.model.batch_size + 1)

        # ── Logging ──────────────────────────────────────────────────────────
        if update % args.log_interval == 0:
            sampled = collect_log.pop("sampled_graphs", [])

            loss_metrics = {
                "global_step": update,
                "loss": total_loss / n_batch_updates,
                "pg_loss": total_pg / n_batch_updates,
                "value_loss": total_vl / n_batch_updates,
                "entropy": total_ent / n_batch_updates,
            }
            logger.log({**loss_metrics, **collect_log})

            if sampled:
                ug = len(set(g[0] for g in sampled))
                print(f"  Graphs: {ug} unique / {len(sampled)} total, "
                      f"last: {sampled[-1][0]} ({sampled[-1][1]} nodes)")

        # ── Checkpoint ───────────────────────────────────────────────────────
        if args.save_path and update % (args.log_interval * 10) == 0:
            _save_checkpoint(args.save_path, model, optimizer, update, args)

    if args.save_path:
        _save_checkpoint(
            args.save_path,
            model,
            optimizer,
            n_updates,
            args,
            final=True,
        )

    return model
