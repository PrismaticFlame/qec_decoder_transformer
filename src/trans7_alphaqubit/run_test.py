#!/usr/bin/env python3
"""
run_test.py - Test evaluation script for trans7 (AlphaQubit on Google data)

Runs the three generalisation tests described in tzu-chen-data.pdf:

  Test-A  (Round extrapolation):   d=3, bX, rounds 17-25
  Test-B  (Distance transfer):     d=5, bX, rounds 1-15  (or available)
  Test-C  (Basis transfer):        d=3, bZ, rounds 1-15

All tests use 50,000 shots per folder from the hardware data:
  events_from_b8.01, meas_from_b8.01, obs_flips_actual.01

Usage:
    python run_test.py --ckpt checkpoints/finetune/finetune_x_d3.pth
    python run_test.py --ckpt <path> --tests A B C
    python run_test.py --ckpt <path> --test A         # single test
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from dataset import SyndromeDataset, MultiRoundDataset, make_loader, load_folder
from layout import get_or_build_layout
from hyperparameters import ModelConfig
from run_pretrain import build_model


# -----------------------------------------------------------------------
# LER computation
# -----------------------------------------------------------------------

@torch.no_grad()
def evaluate_ler(
    model: nn.Module,
    loader,
    device: torch.device,
) -> Dict[str, float]:
    """Return LER and number of shots evaluated."""
    model.eval()
    all_preds, all_labels = [], []
    for batch in tqdm(loader, desc="eval", leave=False):
        batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                 for k, v in batch.items()}
        out = model(batch)
        logits = out["logical_logits"].view(-1)
        labels = batch.get("logical_labels", batch.get("label")).view(-1).float()
        preds = (torch.sigmoid(logits) >= 0.5).float()
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    preds_cat = torch.cat(all_preds)
    labels_cat = torch.cat(all_labels)
    ler = float((preds_cat != labels_cat).float().mean().item())
    return {"ler": ler, "n_shots": len(preds_cat)}


# -----------------------------------------------------------------------
# Test data loader builders
# -----------------------------------------------------------------------

def build_test_loader(
    data_root: Path,
    basis: str,
    distance: int,
    rounds_list: List[int],
    batch_size: int = 256,
    num_workers: int = 0,
) -> tuple:
    """
    Load all available hardware data for (basis, distance, rounds_list)
    into a DataLoader. Returns (loader, n_total_shots).
    """
    prefix = f"surface_code_b{basis.upper()}_d{distance}"
    ds_list: List[SyndromeDataset] = []
    layout_ref = None

    for r in rounds_list:
        folders = sorted(data_root.glob(f"{prefix}_r{r:02d}_center_*"))
        for folder in folders:
            if not (folder / "events_from_b8.01").exists():
                continue
            try:
                layout = get_or_build_layout(folder, distance=distance)
                events, labels, meas = load_folder(folder, layout, prefer_hardware=True)
            except Exception as e:
                print(f"  SKIP {folder.name}: {e}")
                continue

            try:
                ds = SyndromeDataset(events, labels, layout, meas)
                ds_list.append(ds)
                if layout_ref is None:
                    layout_ref = layout
                print(f"  Loaded {folder.name}: {len(ds)} shots, D={events.shape[1]}")
            except Exception as e:
                print(f"  SKIP {folder.name} (dataset error): {e}")

    if not ds_list:
        raise FileNotFoundError(
            f"No hardware test data for b{basis.upper()} d={distance} "
            f"rounds={rounds_list} in {data_root}"
        )

    dataset = MultiRoundDataset(ds_list) if len(ds_list) > 1 else ds_list[0]
    loader = make_loader(dataset, batch_size, shuffle=False,
                         num_workers=num_workers, drop_last=False)
    return loader, len(dataset), layout_ref


# -----------------------------------------------------------------------
# Run individual tests
# -----------------------------------------------------------------------

def run_test_A(
    model: nn.Module,
    data_root: Path,
    device: torch.device,
    batch_size: int = 256,
) -> Dict[str, Any]:
    """Test-A: Round extrapolation (d=3, bX, rounds 17-25)."""
    print("\n--- Test-A: Round Extrapolation (d=3, bX, r17-r25) ---")
    rounds = [17, 19, 21, 23, 25]
    loader, n_shots, layout = build_test_loader(
        data_root, "X", 3, rounds, batch_size=batch_size
    )
    metrics = evaluate_ler(model, loader, device)
    result = {
        "test": "A",
        "description": "Round extrapolation (d=3, bX, rounds 17-25)",
        "basis": "X", "distance": 3, "rounds": rounds,
        "n_shots": n_shots,
        **metrics,
    }
    print(f"  LER = {metrics['ler']:.6f}  ({n_shots} shots)")
    return result


def run_test_B(
    model: nn.Module,
    data_root: Path,
    device: torch.device,
    batch_size: int = 256,
) -> Dict[str, Any]:
    """Test-B: Distance transfer (d=5, bX, rounds 1-25)."""
    print("\n--- Test-B: Distance Transfer (d=5, bX, r1-r25) ---")
    rounds = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
    loader, n_shots, layout = build_test_loader(
        data_root, "X", 5, rounds, batch_size=batch_size
    )
    metrics = evaluate_ler(model, loader, device)
    result = {
        "test": "B",
        "description": "Distance transfer (d=5, bX)",
        "basis": "X", "distance": 5, "rounds": rounds,
        "n_shots": n_shots,
        **metrics,
    }
    print(f"  LER = {metrics['ler']:.6f}  ({n_shots} shots)")
    return result


def run_test_C(
    model: nn.Module,
    data_root: Path,
    device: torch.device,
    batch_size: int = 256,
) -> Dict[str, Any]:
    """Test-C: Basis transfer (d=3, bZ, rounds 1-15)."""
    print("\n--- Test-C: Basis Transfer (d=3, bZ, r1-r15) ---")
    rounds = [1, 3, 5, 7, 9, 11, 13, 15]
    loader, n_shots, layout = build_test_loader(
        data_root, "Z", 3, rounds, batch_size=batch_size
    )
    metrics = evaluate_ler(model, loader, device)
    result = {
        "test": "C",
        "description": "Basis transfer (d=3, bZ)",
        "basis": "Z", "distance": 3, "rounds": rounds,
        "n_shots": n_shots,
        **metrics,
    }
    print(f"  LER = {metrics['ler']:.6f}  ({n_shots} shots)")
    return result


# -----------------------------------------------------------------------
# Checkpoint loading (shared with finetune)
# -----------------------------------------------------------------------

def load_model_from_ckpt(
    ckpt_path: str,
    model_cfg: ModelConfig,
    device: str = "cpu",
) -> tuple:
    """Load model from checkpoint. Returns (model, ckpt_dict)."""
    ckpt = torch.load(ckpt_path, map_location=device)
    layout = ckpt.get("layout")
    basis = ckpt.get("basis", "x")
    if layout is None:
        raise ValueError(
            f"Checkpoint {ckpt_path} does not contain a 'layout' key. "
            "Re-run training with the latest code to save layout in checkpoint."
        )
    model = build_model(layout, model_cfg, basis, use_full_bias=True)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    return model, ckpt


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Trans7 test evaluation: round extrapolation, distance transfer, basis transfer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to fine-tuned (or pre-trained) checkpoint (.pth)")
    parser.add_argument("--tests", nargs="+", choices=["A", "B", "C"],
                        default=["A", "B", "C"],
                        help="Which tests to run")
    parser.add_argument("--data_dir", type=str,
                        default="../../data/trans7_data")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--output", type=str, default=None,
                        help="JSON file to save results (default: next to checkpoint)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device override (default: auto-detect CUDA)")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    data_root = (script_dir / args.data_dir).resolve()
    ckpt_path = Path(args.ckpt)

    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)
    if not data_root.exists():
        print(f"ERROR: data_dir not found: {data_root}", file=sys.stderr)
        sys.exit(1)

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    model_cfg = ModelConfig()
    model_cfg.d_model = args.d_model

    print(f"Loading checkpoint: {ckpt_path}")
    model, ckpt = load_model_from_ckpt(str(ckpt_path), model_cfg, device=device_str)
    model.to(device)
    print(f"  Basis: {ckpt.get('basis', 'unknown')}  "
          f"Distance: {ckpt.get('distance', 'unknown')}  "
          f"Best LER: {ckpt.get('best_ler', 'N/A')}")

    results = []
    test_fns = {"A": run_test_A, "B": run_test_B, "C": run_test_C}

    for test_name in args.tests:
        try:
            result = test_fns[test_name](model, data_root, device, batch_size=args.batch_size)
            results.append(result)
        except FileNotFoundError as e:
            print(f"  SKIPPED Test-{test_name}: {e}")
            results.append({"test": test_name, "error": str(e)})

    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"{'Test':<8} {'Description':<40} {'LER':>10} {'Shots':>8}")
    print(f"{'-'*70}")
    for r in results:
        if "ler" in r:
            print(f"  {r['test']:<6} {r['description']:<40} {r['ler']:>10.6f} {r['n_shots']:>8}")
        else:
            print(f"  {r['test']:<6} SKIPPED: {r.get('error', '')}")
    print(f"{'='*60}")

    # Save results
    out_path = Path(args.output) if args.output else ckpt_path.parent / "test_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
