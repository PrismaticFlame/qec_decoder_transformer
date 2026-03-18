# train.py - Training loop for trans7 (AlphaQubit on Google hardware data)
from __future__ import annotations

import csv
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

try:
    import wandb
except ImportError:
    wandb = None

from dataset import make_loader, set_seed
from ema import EMA
from loss import QECMultiTaskLoss

from utils import Lion

# -----------------------------------------------------------------------
# LR / batch schedule
# -----------------------------------------------------------------------


def get_batch_size(step: int, cfg) -> int:
    bs = cfg.batch_init
    if step >= cfg.batch_change_step:
        bs = min(cfg.batch_final, bs * 2)
    if step >= 2 * cfg.batch_change_step:
        bs = cfg.batch_final
    return bs


def apply_lr_schedule(optimizer: torch.optim.Optimizer, step: int, cfg) -> float:
    k = sum(step >= s for s in cfg.lr_decay_steps)
    lr = cfg.lr * (cfg.lr_decay_factor**k)
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return float(lr)


# -----------------------------------------------------------------------
# LER evaluation
# -----------------------------------------------------------------------


@torch.no_grad()
def compute_ler(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    ema: Optional[EMA] = None,
) -> float:
    """Compute Logical Error Rate (fraction of mis-classified shots)."""
    model.eval()
    if ema is not None:
        ema.apply_to(model)
    try:
        all_preds, all_labels = [], []
        for batch in loader:
            batch = {
                k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            out = model(batch)
            logits = out["logical_logits"].view(-1)
            labels = batch.get("logical_labels", batch.get("label")).view(-1).float()
            preds = (torch.sigmoid(logits) >= 0.5).float()
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
        preds_cat = torch.cat(all_preds)
        labels_cat = torch.cat(all_labels)
        return float((preds_cat != labels_cat).float().mean().item())
    finally:
        if ema is not None:
            ema.restore(model)


# -----------------------------------------------------------------------
# Main training function
# -----------------------------------------------------------------------


def train(
    model: nn.Module,
    train_dataset,
    val_dataset,
    cfg,
    *,
    run_name: str = "trans7_run",
    use_wandb: bool = False,
    checkpoint_path: Optional[str] = None,
    get_next_train_chunk=None,
    rank: int = 0,
    world_size: int = 1,
) -> Dict[str, Any]:
    """
    Train model according to cfg (TrainConfig).

    Args:
        rank: Process rank for distributed training (0 = main process).
        world_size: Total number of processes (1 = single-GPU).

    Returns a dict with best LER, best step, history, and the best checkpoint
    state dict (shadow params from EMA if available).
    """
    is_main = rank == 0
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    model.to(device)

    # Optimizer
    if cfg.optimizer.lower() == "lion":
        optimizer = Lion(
            model.parameters(),
            lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2),
            weight_decay=cfg.weight_decay,
        )
    else:
        raise ValueError(
            f"Unknown optimizer: {cfg.optimizer!r}. Only 'lion' is supported."
        )

    # EMA
    ema = EMA(model, alpha=cfg.ema_alpha) if cfg.use_ema else None

    # AMP — enabled automatically when training on CUDA, no-op on CPU.
    # Use bfloat16 on Ampere+ (A100, H100, RTX 30/40 series) — no overflow,
    # no GradScaler needed. Fall back to float16 on older hardware.
    use_amp = device.type == "cuda"
    if use_amp and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
        use_scaler = False
    else:
        amp_dtype = torch.float16
        use_scaler = use_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    if use_amp and is_main:
        gpu_name = torch.cuda.get_device_name(device)
        print(
            f"  AMP: {amp_dtype} on {gpu_name}"
            + (" (GradScaler off)" if not use_scaler else " (GradScaler on)")
        )

    # Loss criterion
    criterion = QECMultiTaskLoss(
        next_stab_weight=float(getattr(cfg, "next_stab_pred_weight", 0.0)),
        logical_pos_weight=getattr(cfg, "logical_pos_weight", None),
        next_stab_schedule=getattr(cfg, "next_stab_schedule", "none"),
        next_stab_weight_min=float(getattr(cfg, "next_stab_weight_min", 0.0)),
        warmup_ratio=float(getattr(cfg, "next_stab_warmup_ratio", 0.3)),
    ).to(device)

    wb_on = bool(use_wandb and wandb is not None and is_main)
    if wb_on:
        wandb.init(project="alphaqubit-trans7", name=run_name, config=asdict(cfg))

    best: Dict[str, Any] = {
        "step": -1,
        "ler": float("inf"),
        "shadow": None,
    }
    history: Dict[str, List] = {"train": [], "eval": []}

    # Initial loader
    cur_bs = get_batch_size(0, cfg)
    cur_train_dataset = train_dataset
    loader_kwargs = dict(
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        rank=rank,
        world_size=world_size,
    )
    train_loader = make_loader(
        cur_train_dataset,
        cur_bs,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = make_loader(
        val_dataset,
        cur_bs,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    train_iter = iter(train_loader)

    pbar = (
        tqdm(range(cfg.num_steps), desc=run_name, dynamic_ncols=True)
        if is_main
        else range(cfg.num_steps)
    )
    model.train()

    for step in pbar:
        # Batch size schedule
        new_bs = get_batch_size(step, cfg)
        if new_bs != cur_bs:
            cur_bs = new_bs
            train_loader = make_loader(
                cur_train_dataset,
                cur_bs,
                shuffle=True,
                **loader_kwargs,
            )
            val_loader = make_loader(
                val_dataset,
                cur_bs,
                shuffle=False,
                drop_last=False,
                **loader_kwargs,
            )
            train_iter = iter(train_loader)

        lr = apply_lr_schedule(optimizer, step, cfg)

        # Fetch batch
        try:
            batch = next(train_iter)
        except StopIteration:
            if get_next_train_chunk is not None:
                next_chunk = get_next_train_chunk(cur_bs)
                if next_chunk is not None:
                    cur_train_dataset = next_chunk
                    train_loader = make_loader(
                        cur_train_dataset,
                        cur_bs,
                        shuffle=True,
                        **loader_kwargs,
                    )
            train_iter = iter(train_loader)
            batch = next(train_iter)

        batch = {
            k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            out = model(batch)
            logical_logits = out["logical_logits"]
            logical_labels = batch.get("logical_labels", batch.get("label"))
            pred_stabs = out.get("pred_stabs")
            true_stabs = batch.get("true_stabs")
            token_mask = batch.get("token_mask")

            loss, loss_stats = criterion(
                logical_logits=logical_logits,
                logical_labels=logical_labels,
                pred_stabs=pred_stabs,
                true_stabs=true_stabs,
                token_mask=token_mask,
                step=step,
                total_steps=cfg.num_steps,
            )

        scaler.scale(loss).backward()

        if cfg.grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)

        scaler.step(optimizer)
        scaler.update()

        if ema is not None:
            ema.update(model)

        # Logging (rank 0 only)
        if is_main and step % cfg.log_every == 0:
            log = {
                "step": step,
                "lr": lr,
                "batch_size": cur_bs,
                "loss": float(loss.detach().cpu()),
                "loss_main": float(loss_stats["logical_loss"].cpu()),
                "loss_next": float(
                    loss_stats["stab_loss"].cpu()
                    if torch.is_tensor(loss_stats["stab_loss"])
                    else loss_stats["stab_loss"]
                ),
            }
            if wb_on:
                wandb.log(log, step=step)
            pbar.set_postfix(
                {
                    "loss": f"{log['loss']:.4g}",
                    "lr": f"{lr:.3g}",
                    "bs": cur_bs,
                }
            )
            history["train"].append(
                {
                    "step": step,
                    "loss": log["loss"],
                    "loss_main": log["loss_main"],
                    "lr": lr,
                }
            )

        # Validation (rank 0 only)
        if is_main and step > 0 and step % cfg.eval_every == 0:
            dev_ler = compute_ler(model, val_loader, device, ema=ema)

            if wb_on:
                wandb.log({"dev/ler": dev_ler}, step=step)

            if not math.isnan(dev_ler) and dev_ler < best["ler"]:
                best["ler"] = dev_ler
                best["step"] = step
                if ema is not None:
                    best["shadow"] = {
                        k: v.detach().cpu().clone()
                        for k, v in ema.shadow_params.items()
                    }
                else:
                    best["shadow"] = {
                        n: p.detach().cpu().clone()
                        for n, p in model.named_parameters()
                        if p.requires_grad
                    }

                if checkpoint_path is not None:
                    _save_checkpoint(model, ema, cfg, best, checkpoint_path)

            history["eval"].append(
                {
                    "step": step,
                    "dev_ler": dev_ler,
                    "best_ler": best["ler"],
                }
            )
            model.train()

    # Restore best parameters into model
    if best["shadow"] is not None:
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in best["shadow"]:
                    p.copy_(best["shadow"][name].to(device))

    if wb_on:
        wandb.finish()

    best["history"] = history
    return best


# -----------------------------------------------------------------------
# Checkpoint helpers
# -----------------------------------------------------------------------


def _save_checkpoint(model, ema, cfg, best, path):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema.state_dict() if ema is not None else None,
            "best_ler": best["ler"],
            "best_step": best["step"],
        },
        path,
    )


def save_history(history: Dict[str, List], out_dir: Path, run_name: str):
    """Write training/eval history to CSV files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if history.get("train"):
        p = out_dir / f"{run_name}_loss.csv"
        with open(p, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["step", "loss", "loss_main", "lr"])
            writer.writeheader()
            writer.writerows(history["train"])
    if history.get("eval"):
        p = out_dir / f"{run_name}_eval.csv"
        with open(p, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["step", "dev_ler", "best_ler"])
            writer.writeheader()
            writer.writerows(history["eval"])
