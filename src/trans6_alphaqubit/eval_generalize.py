#!/usr/bin/env python3
"""
eval_generalize.py - Evaluate trained QEC decoder on unseen round counts and bases.

Tests two axes of generalization:
  1. Round-count generalization: model trained on r6, tested on r9, r11, r13, r15, r17
  2. Cross-basis generalization: x-basis model tested on z-basis data (and vice versa)

The only weight that depends on round count is the cycle positional embedding
(nn.Embedding(num_cycles, d_model)).  All other weights are round-agnostic.
We adapt the embedding by stretching: first embedding -> first slot, last -> last slot,
middle slots are nearest-neighbor from the trained table.

Usage:
    python eval_generalize.py --checkpoint checkpoints/x_d3_r6.pth
    python eval_generalize.py --checkpoint checkpoints/x_d3_r6.pth \\
        --rounds 6 9 11 13 15 17 27 --test_bases x z
    python eval_generalize.py --checkpoint checkpoints/x_d3_r6.pth \\
        checkpoints/z_d3_r6.pth --rounds 6 9 17 27
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# ── local imports ──────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from model import AlphaQubitLikeModel, ScatteringResidualConvBlock
from dataset import SyndromeDataset
from parameter import ModelConfigScaling
from utils import AttentionBiasProvider, ManhattanDistanceBias
from eval import compute_ler_from_logits


# ═══════════════════════════════════════════════════════════════════════════
# Data utilities
# ═══════════════════════════════════════════════════════════════════════════

def _format_p(p: float) -> str:
    if p >= 0.001:
        return f"{p:.4f}".rstrip("0").rstrip(".")
    return f"{p:.2e}"


def find_existing_data(data_dir: Path, basis: str, distance: int, rounds: int) -> Optional[Path]:
    """Return the newest matching data directory, or None."""
    basis_dir = data_dir / f"{basis}_basis"
    candidates = sorted(basis_dir.glob(f"d{distance}_r{rounds}_p*_s*"))
    if candidates:
        return candidates[-1]
    legacy = basis_dir / f"d{distance}_r{rounds}"
    if legacy.is_dir():
        return legacy
    return None


def generate_data_in_memory(basis: str, distance: int, rounds: int, p: float, shots: int):
    """
    Generate QEC data in memory using stim (no files written).

    Returns (det_hard, meas_hard, obs, layout) or raises ImportError if stim unavailable.
    """
    import stim
    # keep layout import here so we only need stim when generating
    from layout import build_layout_from_circuit

    circuit_name = f"surface_code:rotated_memory_{basis}"
    circ = stim.Circuit.generated(
        circuit_name,
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p,
        before_round_data_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )

    # deterministic seed: avoids collisions across (basis, distance, p, rounds)
    basis_offset = {"z": 0, "x": 10_000}
    seed = 42 + basis_offset[basis] + distance * 100 + round(p * 1e6) + rounds * 10

    sampler = circ.compile_detector_sampler(seed=seed)
    det_events, obs_flips = sampler.sample(shots, separate_observables=True)

    det_hard = det_events.astype(np.int8)
    obs = obs_flips[:, 0:1].astype(np.int8)

    layout = build_layout_from_circuit(circ)
    layout["distance"] = distance

    # reconstruct measurements from detection events
    stab_id_arr = np.array(layout["stab_id"])
    cycle_id_arr = np.array(layout["cycle_id"])
    num_stab = int(stab_id_arr.max()) + 1
    meas_hard = np.zeros_like(det_hard)
    for s in range(num_stab):
        idx = np.where(stab_id_arr == s)[0]
        idx = idx[np.argsort(cycle_id_arr[idx])]
        if len(idx):
            meas_hard[:, idx] = np.cumsum(det_hard[:, idx], axis=1) % 2
    meas_hard = meas_hard.astype(np.int8)

    return det_hard, meas_hard, obs, layout


def get_val_data(data_dir: Path, basis: str, distance: int, rounds: int,
                 p: float, shots: int):
    """
    Return (val_samples, val_labels, val_meas, layout) for (basis, d, rounds).

    Priority:
      1. Existing data directory (uses val.npz + layout.json)
      2. Generate in-memory with stim (validation split = last 20 %)
      3. Return None if neither is possible.
    """
    data_path = find_existing_data(data_dir, basis, distance, rounds)
    if data_path is not None:
        val_npz = np.load(data_path / "val.npz")
        val_samples = val_npz.get("det_soft", val_npz.get("det_hard"))
        val_labels  = val_npz.get("obs", val_npz.get("labels"))
        val_meas    = val_npz.get("meas_hard", None)
        with open(data_path / "layout.json") as f:
            layout = json.load(f)
        if "distance" not in layout:
            layout["distance"] = distance
        print(f"    [data] loaded from {data_path.name}  "
              f"(val n={val_samples.shape[0]})")
        return val_samples, val_labels, val_meas, layout

    # try stim generation
    try:
        det_hard, meas_hard, obs, layout = generate_data_in_memory(
            basis, distance, rounds, p, shots
        )
        n = det_hard.shape[0]
        n_val = int(n * 0.2)
        rng = np.random.RandomState(seed=7)
        idx = rng.permutation(n)
        val_idx = idx[:n_val]
        print(f"    [data] generated with stim  "
              f"(val n={n_val}, basis={basis}, d={distance}, r={rounds})")
        return det_hard[val_idx], obs[val_idx], meas_hard[val_idx], layout
    except ImportError:
        print(f"    [data] MISSING: no data for {basis}_d{distance}_r{rounds} "
              f"and stim is not installed.")
        print(f"           Install with:  pip install stim")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Model building (mirrors run_train.py)
# ═══════════════════════════════════════════════════════════════════════════

def _compute_effective_S(layout: dict) -> int:
    cycle_counts = Counter(layout["cycle_id"])
    return max(cycle_counts.values())


def _build_conv_block(distance: int, d_model: int, layout: dict,
                      model_cfg: ModelConfigScaling) -> ScatteringResidualConvBlock:
    x_coords = np.array(layout["x"], dtype=np.float32)
    y_coords = np.array(layout["y"], dtype=np.float32)
    H = W = distance + 1
    effective_S = _compute_effective_S(layout)

    cycle_counts = Counter(layout["cycle_id"])
    max_cycle = max(cycle_counts, key=cycle_counts.get)
    stab_ids_arr = np.array(layout["stab_id"])
    cycle_ids_arr = np.array(layout["cycle_id"])
    active_stab_ids = sorted([
        stab_ids_arr[i]
        for i in range(len(stab_ids_arr))
        if cycle_ids_arr[i] == max_cycle
    ])[:effective_S]

    num_stab = layout["num_stab"]
    stab_x_all = np.zeros(num_stab, dtype=np.float32)
    stab_y_all = np.zeros(num_stab, dtype=np.float32)
    stab_found = np.zeros(num_stab, dtype=bool)
    for idx in range(len(stab_ids_arr)):
        sid = stab_ids_arr[idx]
        if not stab_found[sid]:
            stab_x_all[sid] = x_coords[idx]
            stab_y_all[sid] = y_coords[idx]
            stab_found[sid] = True

    stab_x = np.array([stab_x_all[sid] for sid in active_stab_ids], dtype=np.float32)
    stab_y = np.array([stab_y_all[sid] for sid in active_stab_ids], dtype=np.float32)

    coord_quant = layout.get("coord_quant", 0.5)
    x_q = np.round(stab_x / coord_quant) * coord_quant
    y_q = np.round(stab_y / coord_quant) * coord_quant
    x_min, x_max = x_q.min(), x_q.max()
    y_min, y_max = y_q.min(), y_q.max()

    x_norm = (x_q - x_min) / (x_max - x_min) * (H - 1) if x_max > x_min else np.zeros_like(x_q)
    y_norm = (y_q - y_min) / (y_max - y_min) * (W - 1) if y_max > y_min else np.zeros_like(y_q)

    i_coords = np.round(x_norm).astype(np.int32).clip(0, H - 1)
    j_coords = np.round(y_norm).astype(np.int32).clip(0, W - 1)

    coord_to_index = {}
    index_to_coord = []
    for k in range(effective_S):
        i, j = int(i_coords[k]), int(j_coords[k])
        coord_to_index[(i, j)] = k
        index_to_coord.append((i, j))

    channels_list = [model_cfg.conv_dim] * model_cfg.conv_layers
    return ScatteringResidualConvBlock(
        d=distance,
        d_d=d_model,
        L_layers=model_cfg.conv_layers,
        channels_list=channels_list,
        coord_to_index=coord_to_index,
        index_to_coord=index_to_coord,
        dilation_list=None,
    )


def build_model_for_layout(layout: dict, model_cfg: ModelConfigScaling,
                            basis: str, use_full_bias: bool = True) -> AlphaQubitLikeModel:
    num_stab  = layout["num_stab"]
    num_cycles = layout["num_cycles"]
    distance  = layout["distance"]
    d_model   = model_cfg.d_model

    conv_blocks = nn.ModuleList([
        _build_conv_block(distance, d_model, layout, model_cfg)
        for _ in range(model_cfg.syndrome_layers)
    ])
    coord_to_index = conv_blocks[0].coord_to_index

    if use_full_bias:
        bias_provider = AttentionBiasProvider(
            db=model_cfg.bias_dim,
            max_dist=8,
            num_residual_layers=model_cfg.bias_residual_layers,
            indicator_features=model_cfg.indicator_features,
            coord_scale=0.5,
        )
    else:
        bias_provider = ManhattanDistanceBias(db=model_cfg.bias_dim, max_dist=8)

    return AlphaQubitLikeModel(
        num_stab=num_stab,
        num_cycles=num_cycles,
        d_model=d_model,
        d_attn=model_cfg.key_size,
        d_mid=model_cfg.key_size,
        db=model_cfg.bias_dim,
        H=model_cfg.num_heads,
        n_layers=model_cfg.syndrome_layers,
        widen=model_cfg.dense_widen,
        conv_blocks=conv_blocks,
        bias_provider=bias_provider,
        use_next_stab=True,
        readout_dim=model_cfg.readout_dim,
        readout_resnet_layers=model_cfg.readout_resnet_layers,
        distance=distance,
        coord_to_index=coord_to_index,
        basis=basis,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Cycle-embedding adaptation
# ═══════════════════════════════════════════════════════════════════════════

def _stretch_embedding(trained: torch.Tensor, test_T: int) -> torch.Tensor:
    """
    Stretch a (train_T, D) cycle embedding to (test_T, D).

    Mapping strategy:
      - Slot 0 in test  -> slot 0 in train  (first measurement)
      - Slot T-1 in test -> slot T-1 in train (final measurement)
      - Middle slots     -> nearest-neighbor interpolation in train
    """
    train_T, D = trained.shape
    if test_T == train_T:
        return trained.clone()

    new_emb = trained.new_zeros(test_T, D)

    if test_T == 1:
        new_emb[0] = trained[0]
        return new_emb

    new_emb[0] = trained[0]          # first preserved
    new_emb[-1] = trained[-1]         # last preserved

    if test_T > 2 and train_T > 2:
        # map middle test slots onto trained middle slots by nearest-neighbor
        for t_new in range(1, test_T - 1):
            frac = (t_new - 1) / max(1, test_T - 2)          # 0..1
            t_train = 1 + frac * (train_T - 2)                 # float index in trained middle
            t_lo = int(t_train)
            t_hi = min(t_lo + 1, train_T - 2)
            alpha = t_train - t_lo
            new_emb[t_new] = (1 - alpha) * trained[t_lo] + alpha * trained[t_hi]

    return new_emb


def adapt_state_dict(state_dict: dict, test_num_cycles: int,
                     n_layers: int = 3) -> dict:
    """
    Return a copy of state_dict adapted for the given num_cycles and model layout.

    Handles two compatibility issues:
      1. Cycle embedding size mismatch  (train_T vs test_T)
      2. Old checkpoints used a single shared  core.conv_block.*
         while current code uses per-layer   core.conv_blocks.{i}.*
         — replicate the single block's weights to all n_layers slots.
    """
    sd = copy.deepcopy(state_dict)

    # ── 1. cycle embedding adaptation ──────────────────────────────────────
    for key in ["embed.cycle_emb.weight", "final_embed.cycle_emb.weight"]:
        if key in sd:
            sd[key] = _stretch_embedding(sd[key], test_num_cycles)

    # ── 2. conv_block → conv_blocks remapping (old → new naming) ──────────
    # Detect old naming by checking for the singular key prefix
    old_prefix = "core.conv_block."
    new_prefix = "core.conv_blocks."
    old_keys = [k for k in sd if k.startswith(old_prefix)]
    if old_keys:
        # replicate single block weights to each layer
        for i in range(n_layers):
            for ok in old_keys:
                suffix = ok[len(old_prefix):]          # e.g. "P" or "lns.0.weight"
                nk = f"{new_prefix}{i}.{suffix}"
                sd[nk] = sd[ok].clone()
        # remove old keys
        for ok in old_keys:
            del sd[ok]

    return sd


# ═══════════════════════════════════════════════════════════════════════════
# MWPM baseline
# ═══════════════════════════════════════════════════════════════════════════

def compute_mwpm_ler(basis: str, distance: int, rounds: int,
                     p: float, shots: int, seed: int = 88888) -> float:
    """Decode with MWPM (PyMatching) and return LER."""
    import stim
    import pymatching

    circ = stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis}",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p,
        before_round_data_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )

    sampler = circ.compile_detector_sampler(seed=seed + rounds * 7)
    det_events, obs_flips = sampler.sample(shots, separate_observables=True)

    dem = circ.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(dem)
    predictions = matching.decode_batch(det_events.astype(np.uint8))

    pred_obs = predictions[:, 0] if predictions.ndim == 2 else predictions
    obs_flat = obs_flips[:, 0]
    return float(np.sum(pred_obs != obs_flat)) / shots


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_on_data(model: nn.Module, val_samples, val_labels, val_meas,
                     layout: dict, device: torch.device,
                     batch_size: int = 256, num_workers: int = 0) -> float:
    """Run model on validation data and return LER."""
    dataset = SyndromeDataset(
        samples=val_samples,
        labels=val_labels,
        layout_json_path=None,
        layout=layout,
        input_mode="hard",
        measurements=val_meas,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=(device.type == "cuda"))

    model.eval()
    all_logits = []
    all_labels = []

    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                 for k, v in batch.items()}
        out = model(batch)
        logits = out["logical_logits"] if isinstance(out, dict) else out
        labels = batch.get("logical_labels", batch.get("label"))
        all_logits.append(logits.view(-1).cpu())
        all_labels.append(labels.view(-1).cpu())

    logits_cat = torch.cat(all_logits)
    labels_cat = torch.cat(all_labels)
    return compute_ler_from_logits(logits_cat, labels_cat)


def load_and_adapt_model(checkpoint_path: Path, layout: dict, basis: str,
                          device: torch.device) -> nn.Module:
    """Load checkpoint and adapt it for the given layout's num_cycles."""
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_cfg = ModelConfigScaling(**ck["model_cfg"])

    model = build_model_for_layout(layout, model_cfg, basis=basis)
    model = model.to(device)

    adapted_sd = adapt_state_dict(
        ck["model_state_dict"],
        layout["num_cycles"],
        n_layers=model_cfg.syndrome_layers,
    )
    model.load_state_dict(adapted_sd, strict=True)
    return model


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════

def plot_training_history(checkpoint_dir: Path, run_names: list[str], output_dir: Path):
    """Plot loss + LER training curves for one or more runs on the same figure."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax_loss, ax_ler = axes
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    any_plotted = False
    for i, name in enumerate(run_names):
        color = colors[i % len(colors)]
        loss_csv = checkpoint_dir / f"{name}_loss.csv"
        eval_csv = checkpoint_dir / f"{name}_eval.csv"

        if loss_csv.exists():
            import csv
            with open(loss_csv) as f:
                rows = list(csv.DictReader(f))
            steps  = [int(r["step"])  for r in rows]
            losses = [float(r["loss"]) for r in rows]

            ax_loss.plot(steps, losses, alpha=0.15, color=color, linewidth=0.5)
            if len(losses) > 20:
                w = min(100, len(losses) // 5)
                smoothed = np.convolve(losses, np.ones(w) / w, mode="valid")
                ax_loss.plot(steps[w - 1:], smoothed, color=color, linewidth=2,
                             label=name)
            else:
                ax_loss.plot(steps, losses, color=color, linewidth=2, label=name)
            any_plotted = True

        if eval_csv.exists():
            import csv
            with open(eval_csv) as f:
                rows = list(csv.DictReader(f))
            esteps = [int(r["step"])    for r in rows]
            lers   = [float(r["dev_ler"]) for r in rows]
            bests  = [float(r["best_ler"]) for r in rows]

            ax_ler.plot(esteps, lers, "o-", color=color, markersize=3,
                        linewidth=1, alpha=0.7, label=f"{name} val LER")
            ax_ler.plot(esteps, bests, "--", color=color, linewidth=1.5,
                        alpha=0.9, label=f"{name} best LER")

    ax_loss.set_xlabel("Step")
    ax_loss.set_ylabel("Training Loss")
    ax_loss.set_title("Training Loss")
    ax_loss.legend(fontsize=8)
    ax_loss.grid(True, alpha=0.3)

    ax_ler.set_xlabel("Step")
    ax_ler.set_ylabel("Validation LER")
    ax_ler.set_title("Validation LER During Training")
    ax_ler.legend(fontsize=8)
    ax_ler.grid(True, alpha=0.3)
    ax_ler.set_yscale("log")

    fig.suptitle("Training History", fontsize=14)
    fig.tight_layout()

    out_path = output_dir / "training_history.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[plot] Training history saved to {out_path}")


def plot_generalization(results: list[dict], output_dir: Path, checkpoint_name: str,
                        mwpm_results: dict | None = None):
    """
    Plot LER vs rounds for different bases, with optional MWPM overlay.

    results: list of dicts with keys: basis, rounds, ler, trained_basis, trained_rounds
    mwpm_results: keyed by (basis, distance) → list of {rounds, ler}
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    markers = {"x": "o", "z": "s"}
    linestyles = {"same": "-", "cross": "--"}

    grouped = {}
    for r in results:
        key = (r["trained_basis"], r["test_basis"])
        grouped.setdefault(key, []).append(r)

    for i, ((tb, xb), pts) in enumerate(sorted(grouped.items())):
        pts = sorted(pts, key=lambda p: p["rounds"])
        rounds = [p["rounds"] for p in pts]
        lers   = [p["ler"]    for p in pts]
        style  = "same" if tb == xb else "cross"
        label  = (f"{checkpoint_name}  "
                  f"{'same basis' if style=='same' else 'cross-basis'} "
                  f"(train {tb.upper()}, test {xb.upper()})")
        ax.plot(rounds, lers,
                marker=markers.get(xb, "^"),
                linestyle=linestyles[style],
                color=colors[i % len(colors)],
                linewidth=2, markersize=8, label=label)

    # ── MWPM overlay ──────────────────────────────────────────────────────
    if mwpm_results:
        dist = results[0].get("distance", 3) if results else 3
        mwpm_colors = {"x": "#555555", "z": "#999999"}
        mwpm_markers = {"x": "^", "z": "v"}
        for basis in sorted({r["test_basis"] for r in results}):
            pts = mwpm_results.get((basis, dist), [])
            if pts:
                pts = sorted(pts, key=lambda p: p["rounds"])
                ax.plot([p["rounds"] for p in pts],
                        [p["ler"]    for p in pts],
                        marker=mwpm_markers.get(basis, "D"),
                        linestyle=":",
                        color=mwpm_colors.get(basis, "black"),
                        linewidth=1.8, markersize=7,
                        label=f"MWPM ({basis.upper()}-basis)")

    # Mark trained round count
    if results:
        trained_r = results[0]["trained_rounds"]
        ax.axvline(x=trained_r, color="gray", linestyle=":", linewidth=1.5,
                   label=f"Trained at r={trained_r}")

    ax.set_xlabel("Round count (r)", fontsize=13)
    ax.set_ylabel("Logical Error Rate (LER)", fontsize=13)
    ax.set_title(f"Round-Count & Cross-Basis Generalization\n{checkpoint_name}", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    out_path = output_dir / f"generalization_{checkpoint_name}.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Generalization curve saved to {out_path}")


def plot_combined_generalization(all_results: dict[str, list[dict]], output_dir: Path):
    """All checkpoints on one figure (cross-basis lines dashed)."""
    if not any(v for v in all_results.values()):
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    ci = 0

    for ck_name, results in sorted(all_results.items()):
        if not results:
            continue
        grouped = {}
        for r in results:
            key = (r["trained_basis"], r["test_basis"])
            grouped.setdefault(key, []).append(r)

        for (tb, xb), pts in sorted(grouped.items()):
            pts = sorted(pts, key=lambda p: p["rounds"])
            rounds = [p["rounds"] for p in pts]
            lers   = [p["ler"]    for p in pts]
            style  = "-" if tb == xb else "--"
            label  = f"{ck_name} → {xb.upper()}-basis"
            ax.plot(rounds, lers, marker="o", linestyle=style,
                    color=colors[ci % len(colors)],
                    linewidth=2, markersize=7, label=label)
            ci += 1

    ax.set_xlabel("Round count (r)", fontsize=13)
    ax.set_ylabel("Logical Error Rate (LER)", fontsize=13)
    ax.set_title("Generalization: LER vs Round Count", fontsize=13)
    ax.legend(fontsize=9, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    fig.tight_layout()
    out_path = output_dir / "generalization_combined.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Combined generalization plot saved to {out_path}")


def _average_by_rounds(pts: list[dict]) -> list[dict]:
    """Average LER values that share the same round count."""
    from collections import defaultdict
    bucket: dict[int, list[float]] = defaultdict(list)
    for p in pts:
        bucket[p["rounds"]].append(p["ler"])
    return [{"rounds": r, "ler": float(np.mean(v))} for r, v in sorted(bucket.items())]


def plot_same_basis_with_mwpm(
    all_results: dict[str, list[dict]],
    mwpm_results: dict[tuple, list[dict]],  # keyed by (basis, distance)
    output_dir: Path,
):
    """
    One PNG per distance: same_basis_vs_mwpm_d3.png, same_basis_vs_mwpm_d5.png, …

    Each plot has two curves:
      • Transformer — average of same-basis models at that distance
      • MWPM        — average of MWPM across bases at that distance
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # group same-basis model results by distance
    from collections import defaultdict
    by_dist: dict[int, list[dict]] = defaultdict(list)
    for results in all_results.values():
        for r in results:
            if r["trained_basis"] == r["test_basis"]:
                by_dist[r["distance"]].append(r)

    for dist in sorted(by_dist.keys()):
        fig, ax = plt.subplots(figsize=(11, 7))

        # Transformer: average X+Z at this distance
        avg_model = _average_by_rounds(by_dist[dist])
        ax.plot([p["rounds"] for p in avg_model],
                [p["ler"]    for p in avg_model],
                marker="o", linestyle="-", color="#1f77b4",
                linewidth=2.5, markersize=9,
                label="Transformer (avg X+Z)")

        # MWPM: average across bases at this distance
        mwpm_pts: list[dict] = []
        for (basis, d), pts in mwpm_results.items():
            if d == dist:
                mwpm_pts.extend(pts)
        if mwpm_pts:
            avg_mwpm = _average_by_rounds(mwpm_pts)
            ax.plot([p["rounds"] for p in avg_mwpm],
                    [p["ler"]    for p in avg_mwpm],
                    marker="s", linestyle="--", color="#d62728",
                    linewidth=2.5, markersize=9,
                    label="MWPM (avg X+Z)")

        ax.set_xlabel("Round count (r)", fontsize=13)
        ax.set_ylabel("Logical Error Rate (LER)", fontsize=13)
        ax.set_title(
            f"Transformer vs MWPM: Round-Count Generalization  (d={dist}, p=0.005)",
            fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

        fig.tight_layout()
        out_path = output_dir / f"same_basis_vs_mwpm_d{dist}.png"
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] Same-basis vs MWPM plot saved to {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate QEC transformer generalization across rounds and bases",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", nargs="+",
                        default=["checkpoints/x_d3_r6.pth"],
                        help="Path(s) to .pth checkpoint file(s)")
    parser.add_argument("--rounds", type=int, nargs="+",
                        default=[6, 9, 11, 13, 15, 17],
                        help="Round counts to test on")
    parser.add_argument("--test_bases", nargs="+", default=["x", "z"],
                        choices=["x", "z"],
                        help="Bases to test on (default: both x and z)")
    parser.add_argument("--distance", type=int, default=3,
                        help="Code distance")
    parser.add_argument("--p", type=float, default=0.005,
                        help="Physical error rate")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Data directory (relative to this script)")
    parser.add_argument("--output_dir", type=str, default="eval_results",
                        help="Output directory for plots and CSV (relative to this script)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory containing checkpoints (for training plots)")
    parser.add_argument("--shots", type=int, default=10_000,
                        help="Shots to generate if data is missing (stim)")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--plot_training", action="store_true", default=True,
                        help="Also plot training history from CSV files")
    parser.add_argument("--no_plot_training", dest="plot_training",
                        action="store_false")
    parser.add_argument("--with_mwpm", action="store_true", default=True,
                        help="Include MWPM baseline (requires stim + pymatching)")
    parser.add_argument("--no_mwpm", dest="with_mwpm", action="store_false")
    parser.add_argument("--mwpm_shots", type=int, default=None,
                        help="Shots for MWPM evaluation (defaults to --shots)")
    args = parser.parse_args()

    script_dir  = _HERE
    data_dir    = (script_dir / args.data_dir).resolve()
    output_dir  = (script_dir / args.output_dir).resolve()
    ck_dir      = (script_dir / args.checkpoint_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # ── training history plots ─────────────────────────────────────────────
    if args.plot_training:
        run_names = [p.stem for p in ck_dir.glob("*.pth")]
        if run_names:
            plot_training_history(ck_dir, sorted(run_names), output_dir)

    # ── generalization evaluation ──────────────────────────────────────────
    all_results: dict[str, list[dict]] = {}

    for ck_arg in args.checkpoint:
        ck_path = (script_dir / ck_arg).resolve()
        if not ck_path.exists():
            print(f"\n[WARN] Checkpoint not found: {ck_path}  — skipping.")
            continue

        ck_name = ck_path.stem
        print(f"\n{'='*60}")
        print(f"Checkpoint: {ck_name}")

        # read metadata without loading model weights yet
        ck_meta = torch.load(ck_path, map_location="cpu", weights_only=False)
        trained_basis  = ck_meta.get("basis", "?")
        trained_rounds = ck_meta.get("rounds", "?")
        trained_ler    = ck_meta.get("best_ler", float("nan"))
        print(f"  Trained basis={trained_basis.upper()}  rounds={trained_rounds}  "
              f"best_train_LER={trained_ler:.5f}")

        # use distance saved in checkpoint; fall back to --distance if absent
        ck_distance = int(ck_meta.get("distance", args.distance))
        print(f"  Distance: d={ck_distance}")

        results: list[dict] = []
        all_results[ck_name] = results

        for test_basis in args.test_bases:
            for rounds in args.rounds:
                print(f"\n  → test basis={test_basis.upper()}  rounds={rounds}")
                data = get_val_data(
                    data_dir, test_basis, ck_distance, rounds,
                    args.p, args.shots,
                )
                if data is None:
                    continue
                val_samples, val_labels, val_meas, layout = data

                try:
                    model = load_and_adapt_model(ck_path, layout, test_basis, device)
                except Exception as e:
                    print(f"    [ERROR] Could not load model: {e}")
                    continue

                ler = evaluate_on_data(
                    model, val_samples, val_labels, val_meas, layout,
                    device, batch_size=args.batch_size,
                )
                print(f"    LER = {ler:.5f}")

                results.append({
                    "checkpoint":     ck_name,
                    "trained_basis":  trained_basis,
                    "trained_rounds": trained_rounds,
                    "distance":       ck_distance,
                    "test_basis":     test_basis,
                    "rounds":         rounds,
                    "ler":            ler,
                })

    # ── combined plot if multiple checkpoints ─────────────────────────────
    if len(args.checkpoint) > 1:
        plot_combined_generalization(all_results, output_dir)

    # ── MWPM baseline ─────────────────────────────────────────────────────
    # keyed by (basis, distance) so d3 and d5 are kept separate
    mwpm_results: dict[tuple, list[dict]] = {}
    if args.with_mwpm:
        mwpm_shots = args.mwpm_shots or args.shots
        # all (basis, distance) pairs actually evaluated — run MWPM for all, not just same-basis
        tested_combos = {
            (r["test_basis"], r["distance"])
            for rows in all_results.values() for r in rows
        }
        try:
            for (basis, dist) in sorted(tested_combos):
                mwpm_results[(basis, dist)] = []
                print(f"\n[MWPM] basis={basis.upper()}  d={dist}")
                for rounds in args.rounds:
                    print(f"  r={rounds}...", end=" ", flush=True)
                    ler = compute_mwpm_ler(basis, dist, rounds, args.p, mwpm_shots)
                    print(f"LER={ler:.5f}")
                    mwpm_results[(basis, dist)].append({"rounds": rounds, "ler": ler})
        except ImportError as e:
            print(f"\n[MWPM] skipped — {e}")

    # ── per-checkpoint plots (with MWPM overlay now available) ───────────
    for ck_name, results in all_results.items():
        if results:
            plot_generalization(results, output_dir, ck_name, mwpm_results)

    # ── same-basis + MWPM combined plot ───────────────────────────────────
    if any(all_results.values()):
        plot_same_basis_with_mwpm(all_results, mwpm_results, output_dir)

    # ── save results CSV ──────────────────────────────────────────────────
    import csv
    all_rows = [r for rows in all_results.values() for r in rows]
    if all_rows:
        csv_path = output_dir / "generalization_results.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\n[csv] Results saved to {csv_path}")

    # ── summary table ─────────────────────────────────────────────────────
    if all_rows:
        print(f"\n{'='*65}")
        print("GENERALIZATION RESULTS")
        print(f"{'='*65}")
        print(f"{'Checkpoint':<22} {'Train':<6} {'Test':<6} {'Rounds':>7} {'LER':>10}")
        print(f"{'-'*65}")
        for r in all_rows:
            print(f"{r['checkpoint']:<22} {r['trained_basis'].upper():<6} "
                  f"{r['test_basis'].upper():<6} {r['rounds']:>7} {r['ler']:>10.5f}")
        print(f"{'='*65}")


if __name__ == "__main__":
    main()
