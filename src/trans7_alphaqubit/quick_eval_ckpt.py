#!/usr/bin/env python3
"""
quick_eval_ckpt.py - Fast LER diagnostic for a pretrain checkpoint.

Uses the val split of pretrain.h5 (no hardware data needed).
Also prints logit distribution to diagnose if the model is outputting
near-0.5 (untrained) or has actually learned something.

Usage:
    cd src/trans7_alphaqubit
    python quick_eval_ckpt.py --ckpt /path/to/pretrain_xz_d3.pth
    python quick_eval_ckpt.py --ckpt /path/to/pretrain_xz_d3.pth --distance 3 --d_model 256
    python quick_eval_ckpt.py --ckpt /path/to/pretrain_xz_d3.pth --h5 ../../data/trans7_data/pretrain.h5
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from dataset_streaming import ChunkedHDF5Dataset, get_reference_layout
from dataset import make_loader
from hyperparameters import ModelConfig
from run_pretrain import build_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device):
    """
    Run model on all batches in loader.
    Returns dict with LER, logit stats, and sigmoid output distribution.
    """
    model.eval()
    all_logits = []
    all_preds = []
    all_labels = []

    for batch in tqdm(loader, desc="evaluating", leave=True):
        batch = {
            k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }
        out = model(batch)
        logits = out["logical_logits"].view(-1).cpu()
        labels = batch.get("logical_labels", batch.get("label")).view(-1).float().cpu()
        preds = (torch.sigmoid(logits) >= 0.5).float()

        all_logits.append(logits)
        all_preds.append(preds)
        all_labels.append(labels)

    logits_cat = torch.cat(all_logits)
    preds_cat  = torch.cat(all_preds)
    labels_cat = torch.cat(all_labels)

    ler = float((preds_cat != labels_cat).float().mean())
    probs = torch.sigmoid(logits_cat).numpy()

    return {
        "ler":           ler,
        "n_shots":       len(labels_cat),
        "label_mean":    float(labels_cat.float().mean()),  # should be ~0.5 for balanced data
        "logit_mean":    float(logits_cat.mean()),
        "logit_std":     float(logits_cat.std()),
        "logit_min":     float(logits_cat.min()),
        "logit_max":     float(logits_cat.max()),
        "prob_near_half": float(((probs > 0.45) & (probs < 0.55)).mean()),  # fraction stuck near 0.5
    }


def load_checkpoint(ckpt_path: str, d_model: int, device: str,
                    h5_fallback: Path = None, distance_fallback: int = 3):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    layout = ckpt.get("layout")
    if layout is None:
        # Intermediate checkpoint from train.py — no layout saved.
        # Fall back to reading layout from pretrain.h5.
        if h5_fallback is None or not h5_fallback.exists():
            raise ValueError(
                "Checkpoint has no 'layout' key (intermediate checkpoint from train.py). "
                "Pass --h5 to point at pretrain.h5 so layout can be read from there."
            )
        dist = ckpt.get("distance", distance_fallback)
        print(f"  No layout in checkpoint — reading layout from {h5_fallback.name} (d={dist})")
        layout = get_reference_layout(h5_fallback, dist)
    basis  = ckpt.get("basis", "x")
    model_cfg = ModelConfig()
    model_cfg.d_model = d_model
    model = build_model(layout, model_cfg, basis, use_full_bias=True)
    state = ckpt.get("model_state_dict", ckpt)
    # torch.compile wraps submodules under _orig_mod — strip that prefix so
    # compiled checkpoints load cleanly into an uncompiled eval model.
    state = {k.replace("._orig_mod.", "."): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  WARNING: missing keys in state_dict: {missing[:5]}")
    return model, ckpt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Quick LER eval on pretrain.h5 val split",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ckpt",     type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--h5",       type=str, default="../../data/trans7_data/pretrain.h5")
    parser.add_argument("--distance", type=int, default=None,
                        help="Distance to eval (default: read from checkpoint)")
    parser.add_argument("--d_model",  type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--chunk_size", type=int, default=10_000,
                        help="Val samples to load (larger = more reliable LER estimate)")
    parser.add_argument("--device",   type=str, default=None)
    args = parser.parse_args()

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    print(f"Device: {device}")

    ckpt_path = Path(args.ckpt)
    h5_path   = (Path(__file__).parent / args.h5).resolve()

    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)
    if not h5_path.exists():
        print(f"ERROR: pretrain.h5 not found: {h5_path}", file=sys.stderr)
        sys.exit(1)

    # --- Load model ---
    print(f"\nLoading checkpoint: {ckpt_path.name}")
    model, ckpt = load_checkpoint(str(ckpt_path), args.d_model, device_str,
                                  h5_fallback=h5_path,
                                  distance_fallback=args.distance or 3)
    model.to(device)

    ckpt_distance  = ckpt.get("distance", "?")
    ckpt_best_ler  = ckpt.get("best_ler",  "N/A")
    ckpt_best_step = ckpt.get("best_step", "N/A")
    ckpt_basis     = ckpt.get("basis",     "?")
    print(f"  Checkpoint info: basis={ckpt_basis}  d={ckpt_distance}  "
          f"best_ler={ckpt_best_ler}  best_step={ckpt_best_step}")

    distance = args.distance or (ckpt_distance if isinstance(ckpt_distance, int) else 3)
    print(f"  Evaluating at distance={distance}")

    # --- Load val data from pretrain.h5 ---
    print(f"\nLoading val split from: {h5_path.name}")
    val_ds = ChunkedHDF5Dataset(
        h5_path,
        split="val",
        chunk_size=args.chunk_size,
        distance=distance,
        shuffle=False,
    )
    val_chunk = val_ds.load_chunk(0)
    loader = make_loader(val_chunk, batch_size=args.batch_size,
                         shuffle=False, num_workers=0, drop_last=False)
    print(f"  Val samples loaded: {len(val_chunk)}")

    # --- Evaluate ---
    print()
    metrics = evaluate(model, loader, device)

    # --- Print results ---
    print(f"\n{'='*55}")
    print(f"  LER:            {metrics['ler']:.6f}   ({metrics['n_shots']:,} shots)")
    print(f"  Random baseline: 0.500000")
    print(f"{'='*55}")
    print(f"\n  Logit distribution (raw model output before sigmoid):")
    print(f"    mean:  {metrics['logit_mean']:+.4f}   (0.0 = sigmoid → 0.5 = untrained)")
    print(f"    std:   {metrics['logit_std']:.4f}    (low std = low confidence on all samples)")
    print(f"    range: [{metrics['logit_min']:+.4f}, {metrics['logit_max']:+.4f}]")
    print(f"\n  Fraction of predictions stuck near p=0.5: {metrics['prob_near_half']:.1%}")
    print(f"  Label balance (should be ~0.5):           {metrics['label_mean']:.4f}")
    print()

    # --- Diagnosis ---
    print("  DIAGNOSIS:")
    if metrics["ler"] > 0.48:
        print("  !! LER near 0.5 — model is performing at random chance.")
        if metrics["prob_near_half"] > 0.5:
            print("  !! >50% of logits near 0 — model is outputting near-uniform probabilities.")
            print("     This usually means: learning rate is too high/low, or training")
            print("     is not converging due to an optimization or data issue.")
        else:
            print("  !! Logits have spread but still wrong — possible label/data mismatch.")
    elif metrics["ler"] > 0.30:
        print("  ~ LER above 0.30 — model has learned something but is still poor.")
        print("    This is early-stage learning. Check if loss is decreasing in the .out log.")
    else:
        print(f"  OK  LER = {metrics['ler']:.4f} — model has learned meaningfully.")


if __name__ == "__main__":
    main()
