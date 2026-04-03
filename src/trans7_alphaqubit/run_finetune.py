#!/usr/bin/env python3
"""
run_finetune.py - Fine-tuning entry point for trans7 (AlphaQubit on Google data)

Fine-tuning uses actual hardware data from the .01 files:
  events_from_b8.01    - detection events (50k shots per folder)
  meas_from_b8.01      - measurements (50k shots, when available)
  obs_flips_actual.01  - actual logical error labels (50k shots)

Fine-tuning configuration (from tzu-chen-data.pdf):
  - Distance: d=3
  - Rounds: r1 through r15  (rounds=1,3,5,...,15 in the data_0301 naming)
  - Basis: bX
  - Split: 40,000 train / 10,000 val  (of 50k shots per folder)
  - Primary loss: obs_flips_actual.01

The script loads a pre-trained checkpoint and continues training with the
hardware data and (optionally) a stronger weight decay.

Usage:
    python run_finetune.py --pretrain_ckpt checkpoints/pretrain/pretrain_x_d3.pth
    python run_finetune.py --pretrain_ckpt <path> --num_steps 200000 --eval_every 2000
    python run_finetune.py --pretrain_ckpt <path> --basis z  # fine-tune for Z basis

Checkpoints saved to:
    checkpoints/finetune/{basis}_d{distance}_finetune.pth
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from dataset import (
    SyndromeDataset, MultiRoundDataset,
    load_folder, make_train_val_split,
)
from layout import get_or_build_layout
from hyperparameters import ModelConfig, TrainConfig, get_lr, get_dilations, finetune_config
from train import train, save_history
from run_pretrain import build_model, build_conv_blocks  # reuse model builder


# -----------------------------------------------------------------------
# Folder discovery for fine-tuning
# -----------------------------------------------------------------------

def find_finetune_folders(
    data_root: Path,
    bases: List[str],
    distance: int,
    rounds_list: List[int],
) -> List[Path]:
    """Find hardware data folders for all specified bases and distance."""
    folders = []
    for basis in bases:
        prefix = f"surface_code_b{basis.upper()}_d{distance}"
        for r in rounds_list:
            matches = sorted(data_root.glob(f"{prefix}_r{r:02d}_center_*"))
            for m in matches:
                if (m / "events_from_b8.01").exists():
                    folders.append(m)
    return folders


def load_finetune_datasets(
    data_root: Path,
    bases: List[str],
    distance: int,
    rounds_list: List[int],
    *,
    n_train: int = 40_000,
    n_val: int = 10_000,
    seed: int = 42,
) -> tuple:
    """
    Build aggregated train/val SyndromeDatasets for fine-tuning.

    Loads all specified bases combined so the model learns basis-agnostic representations.
    prefer_hardware=True so events_from_b8.01 / obs_flips_actual.01 are used.
    Returns (train_dataset, val_dataset, layout)
    """
    folders = find_finetune_folders(data_root, bases, distance, rounds_list)
    if not folders:
        raise FileNotFoundError(
            f"No hardware data folders for bases={bases} d={distance} "
            f"rounds={rounds_list} in {data_root}"
        )
    bases_str = "+".join(b.upper() for b in sorted(bases))
    print(f"  Found {len(folders)} folder(s) for bases={bases_str} d={distance} (fine-tuning)")

    train_ds_list: List[SyndromeDataset] = []
    val_ds_list: List[SyndromeDataset] = []
    max_round = max(rounds_list)
    layout_ref = None

    for folder in folders:
        try:
            layout = get_or_build_layout(folder, distance=distance)
        except FileNotFoundError as e:
            print(f"    SKIP {folder.name}: {e}")
            continue
        # Use max-round layout as reference so num_stab covers all stabilizer positions
        if layout_ref is None or (
            f"_r{max_round:02d}_" in folder.name and layout["num_stab"] > layout_ref["num_stab"]
        ):
            layout_ref = layout

        try:
            events, labels, meas = load_folder(folder, layout, prefer_hardware=True)
        except FileNotFoundError as e:
            print(f"    SKIP {folder.name}: {e}")
            continue

        N = events.shape[0]
        actual_train = min(n_train, int(N * 0.8))
        actual_val = min(n_val, N - actual_train)

        (ev_tr, lb_tr, ms_tr), (ev_val, lb_val, ms_val) = make_train_val_split(
            events, labels, meas,
            n_train=actual_train, n_val=actual_val, seed=seed,
        )

        try:
            train_ds_list.append(SyndromeDataset(ev_tr, lb_tr, layout, ms_tr))
            val_ds_list.append(SyndromeDataset(ev_val, lb_val, layout, ms_val))
            print(f"    Loaded {folder.name}: train={actual_train}, val={actual_val}, "
                  f"D={events.shape[1]}")
        except Exception as e:
            print(f"    SKIP {folder.name}: {e}")

    if not train_ds_list:
        raise RuntimeError(
            f"No usable fine-tuning data for bases={bases} d={distance}."
        )

    train_dataset = MultiRoundDataset(train_ds_list) if len(train_ds_list) > 1 else train_ds_list[0]
    val_dataset = MultiRoundDataset(val_ds_list) if len(val_ds_list) > 1 else val_ds_list[0]
    return train_dataset, val_dataset, layout_ref


# -----------------------------------------------------------------------
# Checkpoint loading
# -----------------------------------------------------------------------

def load_pretrained(
    ckpt_path: str,
    layout: Dict[str, Any],
    model_cfg: ModelConfig,
    basis: str,
    use_full_bias: bool = True,
    device: str = "cpu",
) -> nn.Module:
    """
    Load a pre-trained checkpoint, rebuilding the model architecture from the
    checkpoint's stored layout (or the provided one).
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    # Use layout from checkpoint if available (more reliable for architecture)
    ckpt_layout = ckpt.get("layout", layout)

    # If the checkpoint was for a different basis, we still use the same architecture
    # but the readout is basis-specific — so we rebuild with the target basis.
    model = build_model(ckpt_layout, model_cfg, basis, use_full_bias=use_full_bias)
    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    return model


# -----------------------------------------------------------------------
# Fine-tuning entry point
# -----------------------------------------------------------------------

def finetune_single(
    bases: List[str],
    distance: int,
    data_root: Path,
    checkpoint_dir: Path,
    rounds_list: List[int],
    pretrain_ckpt: Optional[str],
    train_cfg: Optional[TrainConfig] = None,
    model_cfg: Optional[ModelConfig] = None,
    use_full_bias: bool = True,
    use_wandb: bool = False,
) -> Dict[str, Any]:
    """Fine-tune one model for a given distance on all specified bases combined."""
    if train_cfg is None:
        train_cfg = finetune_config(distance)
    if model_cfg is None:
        model_cfg = ModelConfig()

    bases_str = "+".join(b.upper() for b in sorted(bases))
    print(f"\n{'='*60}")
    print(f"FINETUNE  bases={bases_str}  d={distance}  rounds={rounds_list}")
    if pretrain_ckpt:
        print(f"  Pretrained ckpt: {pretrain_ckpt}")
    print(f"{'='*60}")

    train_dataset, val_dataset, layout = load_finetune_datasets(
        data_root, bases, distance, rounds_list,
        n_train=40_000, n_val=10_000, seed=train_cfg.seed,
    )

    if pretrain_ckpt is not None and Path(pretrain_ckpt).exists():
        model = load_pretrained(
            pretrain_ckpt, layout, model_cfg, "x",
            use_full_bias=use_full_bias, device=train_cfg.device,
        )
        print(f"  Loaded pretrained weights from {pretrain_ckpt}")
    else:
        if pretrain_ckpt:
            print(f"  WARNING: pretrain ckpt not found ({pretrain_ckpt}), training from scratch.")
        model = build_model(layout, model_cfg, "x", use_full_bias=use_full_bias)

    # Apply fine-tuning weight decay if configured
    if train_cfg.fine_tuning_weight_decay is not None:
        for pg in [{"params": model.parameters(),
                    "weight_decay": train_cfg.fine_tuning_weight_decay}]:
            pass  # Will be used when constructing the optimizer via train_cfg

    n_params = sum(p.numel() for p in model.parameters())
    device = torch.device(train_cfg.device)
    model.to(device)
    print(f"  Parameters: {n_params:,}   Device: {device}")
    print(f"  Train shots: {len(train_dataset)}  Val shots: {len(val_dataset)}")

    run_name = f"finetune_{bases_str.lower()}_d{distance}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = str(checkpoint_dir / f"{run_name}.pth")

    t0 = time.time()
    best = train(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        cfg=train_cfg,
        run_name=run_name,
        use_wandb=use_wandb,
        checkpoint_path=ckpt_path,
    )
    elapsed = time.time() - t0

    torch.save({
        "model_state_dict": model.state_dict(),
        "layout": layout,
        "bases": bases,
        "distance": distance,
        "best_ler": best["ler"],
        "best_step": best["step"],
        "train_time": elapsed,
        "pretrain_ckpt": pretrain_ckpt,
    }, ckpt_path)
    print(f"\n  Best LER: {best['ler']:.6f} at step {best['step']}")
    print(f"  Checkpoint: {ckpt_path}")

    if best.get("history"):
        save_history(best["history"], checkpoint_dir, run_name)

    return {
        "bases": bases, "distance": distance,
        "best_ler": best["ler"], "best_step": best["step"],
        "train_time": elapsed, "checkpoint_path": ckpt_path,
    }


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Trans7 fine-tuning on Google hardware data. "
                    "Trains one model per distance on all bases combined.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pretrain_ckpt", type=str, default=None,
                        help="Path to pre-trained checkpoint (.pth)")
    parser.add_argument("--bases", nargs="+", choices=["x", "z"], default=["x", "z"],
                        help="Bases to include in fine-tuning (combined into one model)")
    parser.add_argument("--distance", type=int, default=3)
    parser.add_argument("--rounds", type=int, nargs="+",
                        default=[1, 3, 5, 7, 9, 11, 13, 15],
                        help="Round counts to use for fine-tuning (r1-r15 per plan)")
    parser.add_argument("--data_dir", type=str,
                        default="../../data/trans7_data")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/finetune")
    parser.add_argument("--num_steps", type=int, default=200_000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--eval_every", type=int, default=2_000)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--no_full_bias", action="store_true")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    data_root = (script_dir / args.data_dir).resolve()
    checkpoint_dir = (script_dir / args.checkpoint_dir).resolve()

    if not data_root.exists():
        print(f"ERROR: data_dir not found: {data_root}", file=sys.stderr)
        sys.exit(1)

    cfg = finetune_config(args.distance)
    cfg.num_steps = args.num_steps
    cfg.batch_init = args.batch_size
    cfg.eval_every = args.eval_every
    if args.lr is not None:
        cfg.lr = args.lr

    model_cfg = ModelConfig()
    model_cfg.d_model = args.d_model

    result = finetune_single(
        bases=args.bases,
        distance=args.distance,
        data_root=data_root,
        checkpoint_dir=checkpoint_dir,
        rounds_list=args.rounds,
        pretrain_ckpt=args.pretrain_ckpt,
        train_cfg=cfg,
        model_cfg=model_cfg,
        use_full_bias=not args.no_full_bias,
        use_wandb=args.use_wandb,
    )

    print(f"\nDone. Best LER: {result['best_ler']:.6f}  "
          f"at step {result['best_step']}")


if __name__ == "__main__":
    main()
