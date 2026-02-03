# train.py
from __future__ import annotations

import math
from dataclasses import asdict
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm.auto import tqdm

try:
    import wandb
except Exception:
    wandb = None

from parameter import ScalingConfig
from dataset import make_loader, set_seed
from ema import EMA
from utils import Lion
# >>> CHANGED: import evaluation functions from eval.py
from eval import compute_ler_from_logits, compute_cycle_ler, evaluate_ler_with_fit

# >>> CHANGED: import criterion
from loss import QECMultiTaskLoss


# -----------------------------
# Schedules: batch + lr
# -----------------------------
def get_batch_size(step: int, cfg: ScalingConfig) -> int:
    bs = cfg.batch_init
    if step >= cfg.batch_change_step:
        bs = min(cfg.batch_final, bs * 2)
    if step >= 2 * cfg.batch_change_step:
        bs = cfg.batch_final
    return bs


def apply_lr_schedule(optimizer: torch.optim.Optimizer, step: int, cfg: ScalingConfig) -> float:
    k = sum(step >= s for s in cfg.lr_decay_steps)
    lr = cfg.lr * (cfg.lr_decay_factor ** k)
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return float(lr)


# Evaluation functions moved to eval.py
# Imported at top of file


# -----------------------------
# Training loop (tqdm + optional wandb)
# -----------------------------
def train(
    model: nn.Module,
    train_dataset,
    val_dataset,
    cfg: ScalingConfig,
    run_name: str = "scaling_run",
    use_wandb: bool = True,
):
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
        raise ValueError("Only lion is wired in this snippet.")

    # EMA
    ema = EMA(model, alpha=cfg.ema_alpha) if cfg.use_ema else None

    # >>> CHANGED: build criterion ONCE (moves to device)
    criterion = QECMultiTaskLoss(
        next_stab_weight=float(getattr(cfg, "next_stab_pred_weight", 0.0)),
        logical_pos_weight=getattr(cfg, "logical_pos_weight", None),
        stab_pos_weight=getattr(cfg, "stab_pos_weight", None),
        # schedule is optional if your loss.py supports it; defaults to "none"
        next_stab_schedule=getattr(cfg, "next_stab_schedule", "none"),
        next_stab_weight_min=float(getattr(cfg, "next_stab_weight_min", 0.0)),
        warmup_ratio=float(getattr(cfg, "next_stab_warmup_ratio", 0.3)),
        decay_ratio=float(getattr(cfg, "next_stab_decay_ratio", 0.5)),
    ).to(device)

    best = {
        "step": -1,
        "ler": float("inf"),
        "shadow": None,
        "fit_r2": None,
        "fit_intercept": None,
    }

    # Accumulate training history for post-hoc analysis / plotting
    history = {"train": [], "eval": []}

    wb_on = bool(use_wandb and (wandb is not None))
    if wb_on:
        wandb.init(project="alphaqubit-scaling", name=run_name, config=asdict(cfg))

    cur_bs = get_batch_size(0, cfg)
    train_loader = make_loader(train_dataset, cur_bs, cfg, shuffle=True)
    val_loader = make_loader(val_dataset, cur_bs, cfg, shuffle=False)
    train_iter = iter(train_loader)

    pbar = tqdm(range(cfg.num_steps), desc="train", dynamic_ncols=True)
    model.train()

    for step in pbar:
        new_bs = get_batch_size(step, cfg)
        if new_bs != cur_bs:
            cur_bs = new_bs
            train_loader = make_loader(train_dataset, cur_bs, cfg, shuffle=True)
            val_loader = make_loader(val_dataset, cur_bs, cfg, shuffle=False)
            train_iter = iter(train_loader)

        lr = apply_lr_schedule(optimizer, step, cfg)

        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        if isinstance(batch, dict):
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            batch = [v.to(device, non_blocking=True) if torch.is_tensor(v) else v for v in batch]
        else:
            raise TypeError("Batch must be dict or list/tuple.")

        optimizer.zero_grad(set_to_none=True)

        # >>> CHANGED: model returns logits only
        out = model(batch)

        # 優先使用分別的 X/Z 模式
        logical_logits_x = out.get("logical_logits_x", None)
        logical_logits_z = out.get("logical_logits_z", None)
        label_x = batch.get("label_x", None)
        label_z = batch.get("label_z", None)

        # logical head (preferred key) - 向後兼容
        logical_logits = out.get("logical_logits", out.get("logits", None))
        logical_labels = batch.get("logical_labels", batch.get("label", None))

        # optional stab head
        pred_stabs = out.get("pred_stabs", None)
        true_stabs = batch.get("true_stabs", None)
        token_mask = batch.get("token_mask", None)

        # optional override: hard-disable next-stab after some step (keep your old behavior)
        # If you prefer schedule-only, you can remove this block.
        if hasattr(cfg, "disable_next_stab_step") and cfg.disable_next_stab_step is not None:
            if step >= int(cfg.disable_next_stab_step):
                pred_stabs = None
                true_stabs = None
                token_mask = None

        # 優先使用分別的 X/Z 模式
        if logical_logits_x is not None and logical_logits_z is not None and label_x is not None and label_z is not None:
            loss, loss_stats = criterion(
                logical_logits_x=logical_logits_x,
                logical_logits_z=logical_logits_z,
                label_x=label_x,
                label_z=label_z,
                pred_stabs=pred_stabs,
                true_stabs=true_stabs,
                token_mask=token_mask,
                step=step,
                total_steps=int(cfg.num_steps),
            )
        elif logical_logits is not None and logical_labels is not None:
            # 向後兼容模式
            loss, loss_stats = criterion(
                logical_logits=logical_logits,
                logical_labels=logical_labels,
                pred_stabs=pred_stabs,
                true_stabs=true_stabs,
                token_mask=token_mask,
                step=step,
                total_steps=int(cfg.num_steps),
            )
        else:
            raise KeyError(
                "Must provide either (logical_logits, logical_labels) "
                "or (logical_logits_x, logical_logits_z, label_x, label_z)"
            )
        loss.backward()

        if cfg.grad_clip_norm is not None and cfg.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)

        optimizer.step()

        if ema is not None:
            ema.update(model)

        # logging
        if step % cfg.log_every == 0:
            # loss_stats values are detached already
            loss_main = loss_stats["logical_loss"]
            loss_next = loss_stats["stab_loss"]
            next_w = loss_stats.get("next_stab_weight", float(getattr(cfg, "next_stab_pred_weight", 0.0)))
            
            # X 和 Z 的損失（如果可用）
            loss_x = loss_stats.get("logical_loss_x", None)
            loss_z = loss_stats.get("logical_loss_z", None)

            log = {
                "step": step,
                "lr": lr,
                "batch_size": cur_bs,
                "loss": float(loss.detach().cpu()),
                "loss_main": float(loss_main.cpu()) if torch.is_tensor(loss_main) else float(loss_main),
                "loss_next": float(loss_next.cpu()) if torch.is_tensor(loss_next) else float(loss_next),
                "next_stab_w": float(next_w),
            }
            
            # 如果可用，添加 X 和 Z 的損失
            if loss_x is not None:
                log["loss_x"] = float(loss_x.cpu()) if torch.is_tensor(loss_x) else float(loss_x)
            if loss_z is not None:
                log["loss_z"] = float(loss_z.cpu()) if torch.is_tensor(loss_z) else float(loss_z)
            if wb_on:
                wandb.log(log, step=step)

            pbar.set_postfix({
                "loss": f"{log['loss']:.4g}",
                "lr": f"{log['lr']:.3g}",
                "bs": cur_bs,
            })

            history["train"].append({
                "step": step,
                "loss": log["loss"],
                "loss_main": log["loss_main"],
                "lr": lr,
            })

        # dev eval (unchanged)
        if (step > 0) and (step % cfg.eval_every == 0):
            eval_fit_mode = getattr(cfg, "eval_fit_mode", "sycamore").lower()

            if eval_fit_mode == "sycamore":
                cycles = list(getattr(cfg, "eval_cycles", (3, 5, 7, 9, 11, 13, 15, 25)))
                metrics = evaluate_ler_with_fit(
                    model=model,
                    val_loader=val_loader,
                    device=device,
                    ema=ema,
                    cycles_for_fit=cycles,
                    min_r2=float(cfg.min_r2),
                    min_intercept=float(cfg.min_intercept),
                    use_ema=True,
                )
                dev_ler = float(metrics["dev/ler_pred25"])
                fit_ok = bool(metrics["dev/fit_ok"] > 0.5)

            else:
                cycles = [25]
                ler_by_cycle = compute_cycle_ler(
                    model, val_loader, device, cycles,
                    use_ema=True, ema=ema
                )
                dev_ler = float(ler_by_cycle[25])
                metrics = {"dev/ler_cycle_25": dev_ler, "dev/fit_ok": 1.0}
                fit_ok = True

            if wb_on:
                wandb.log(metrics, step=step)

            if fit_ok and (not math.isnan(dev_ler)) and dev_ler < best["ler"]:
                best["ler"] = dev_ler
                best["step"] = step
                best["fit_r2"] = metrics.get("dev/fit_r2", None)
                best["fit_intercept"] = metrics.get("dev/fit_intercept", None)

                if ema is not None:
                    best["shadow"] = {k: v.detach().cpu().clone() for k, v in ema.shadow_params.items()}
                else:
                    best["shadow"] = {
                        name: p.detach().cpu().clone()
                        for name, p in model.named_parameters()
                        if p.requires_grad
                    }

                if wb_on:
                    wandb.log({"best/ler": best["ler"], "best/step": best["step"]}, step=step)

            history["eval"].append({
                "step": step,
                "dev_ler": dev_ler,
                "fit_ok": fit_ok,
                "best_ler": best["ler"],
            })

            model.train()

    # restore best into model at end
    if best["shadow"] is not None:
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in best["shadow"]:
                    p.copy_(best["shadow"][name].to(device))

    if wb_on:
        wandb.log({"best/final_ler": best["ler"], "best/final_step": best["step"]}, step=cfg.num_steps)
        wandb.finish()

    best["history"] = history
    return best
