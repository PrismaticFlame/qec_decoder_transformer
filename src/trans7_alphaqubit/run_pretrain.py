#!/usr/bin/env python3
"""
run_pretrain.py - Pretraining entry point for trans7 (AlphaQubit on Google data)

Pretraining uses DEM-simulated data from the .01 files:
  events.01       - detection events  (20k shots per folder)
  meas.01         - measurements      (20k shots per folder, fallback: cumXOR)
  obs_flips.01    - logical labels    (DEM-based, 20k shots)

Data sources:
  - Distances: d=3 and d=5
  - Rounds: 1, 3, 5, ..., 25  (odd rounds from the data_0301 folder)
  - Bases: bX and bZ
  - Splits: 16000 train / 4000 val  (80/20 of 20k)

Each (basis, distance, rounds, center) folder is loaded independently and
combined into a MultiRoundDataset, so the model sees all round lengths during
a single training run.

Usage:
    python run_pretrain.py
    python run_pretrain.py --distances 3 5 --num_steps 500000
    python run_pretrain.py --basis x --distance 3 --num_steps 100000 --eval_every 1000

Checkpoints saved to:
    checkpoints/pretrain/{basis}_d{distance}.pth
"""

import argparse
import json
import queue
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from dataset import (
    MultiRoundDataset,
    SyndromeDataset,
    load_folder,
    make_loader,
    make_train_val_split,
)
from hyperparameters import (
    ModelConfig,
    TrainConfig,
    get_dilations,
    get_lr,
    pretrain_config,
)
from layout import get_or_build_layout
from model import AlphaQubitModel, ScatteringResidualConvBlock

from train import save_history, train
from utils import AttentionBiasProviderUnified, AttentionBiasProviderSplit, ManhattanDistanceBias

# -----------------------------------------------------------------------
# Folder discovery helpers
# -----------------------------------------------------------------------


def find_folders(
    data_root: Path,
    bases: List[str],
    distance: int,
    rounds_list: List[int],
) -> List[Path]:
    """Find all data folders for (bases, distance) across the given round counts."""
    folders = []
    for basis in bases:
        prefix_base = f"surface_code_b{basis.upper()}_d{distance}"
        for r in rounds_list:
            pattern = f"{prefix_base}_r{r:02d}_center_*"
            matches = sorted(data_root.glob(pattern))
            folders.extend(matches)
    return folders


def load_pretraining_datasets(
    data_root: Path,
    bases: List[str],
    distance: int,
    rounds_list: List[int],
    *,
    n_train: int = 16_000,
    n_val: int = 4_000,
    seed: int = 42,
) -> tuple:
    """
    Build aggregated train/val SyndromeDatasets for pretraining.

    Loads all bases (X and Z) for the given distance into a single combined dataset
    so the model learns basis-agnostic representations. One model is trained per distance.

    Returns (train_dataset, val_dataset, layout)
    """
    folders = find_folders(data_root, bases, distance, rounds_list)
    if not folders:
        raise FileNotFoundError(
            f"No data folders found for bases={bases} d={distance} "
            f"rounds={rounds_list} in {data_root}"
        )

    print(
        f"  Found {len(folders)} folder(s) for bases={[b.upper() for b in bases]} d={distance}"
    )

    train_ds_list: List[SyndromeDataset] = []
    val_ds_list: List[SyndromeDataset] = []
    # Use the layout from the max-round folder as the model reference.
    # Longer circuits see all stabilizer positions (both X- and Z-type),
    # so num_stab is largest there. r01 only sees on-basis stabs.
    max_round = max(rounds_list)
    layout_ref = None

    for folder in folders:
        try:
            layout = get_or_build_layout(folder, distance=distance)
        except FileNotFoundError as e:
            print(f"    SKIP {folder.name}: {e}")
            continue

        # Pick layout from a max-round folder as the model reference
        if layout_ref is None or (
            f"_r{max_round:02d}_" in folder.name
            and layout["num_stab"] > layout_ref["num_stab"]
        ):
            layout_ref = layout

        try:
            events, labels, meas = load_folder(folder, layout, prefer_hardware=False)
        except FileNotFoundError as e:
            print(f"    SKIP {folder.name}: {e}")
            continue

        N = events.shape[0]
        actual_n_train = min(n_train, int(N * 0.8))
        actual_n_val = min(n_val, N - actual_n_train)

        (ev_tr, lb_tr, ms_tr), (ev_val, lb_val, ms_val) = make_train_val_split(
            events,
            labels,
            meas,
            n_train=actual_n_train,
            n_val=actual_n_val,
            seed=seed,
        )

        try:
            train_ds_list.append(SyndromeDataset(ev_tr, lb_tr, layout, ms_tr))
            val_ds_list.append(SyndromeDataset(ev_val, lb_val, layout, ms_val))
            print(
                f"    Loaded {folder.name}: train={actual_n_train}, val={actual_n_val}, "
                f"D={events.shape[1]}"
            )
        except Exception as e:
            print(f"    SKIP {folder.name}: dataset error: {e}")
            continue

    if not train_ds_list:
        raise RuntimeError(
            f"No usable data for bases={bases} d={distance}. "
            "Check that events.01 and obs_flips.01 exist in the data folders."
        )

    train_dataset = (
        MultiRoundDataset(train_ds_list) if len(train_ds_list) > 1 else train_ds_list[0]
    )
    val_dataset = (
        MultiRoundDataset(val_ds_list) if len(val_ds_list) > 1 else val_ds_list[0]
    )

    return train_dataset, val_dataset, layout_ref


# -----------------------------------------------------------------------
# Model construction
# -----------------------------------------------------------------------


def build_conv_blocks(
    distance: int,
    d_model: int,
    layout: Dict[str, Any],
    model_cfg: ModelConfig,
) -> nn.ModuleList:
    """Build per-layer ScatteringResidualConvBlocks with correct dilations."""
    import numpy as np

    x_coords = np.array(layout["x"], dtype=np.float32)
    y_coords = np.array(layout["y"], dtype=np.float32)

    stab_ids_arr = np.array(layout["stab_id"])
    num_stab = layout["num_stab"]

    # Collect the unique (x, y) position for each stabilizer ID.
    stab_x_all = np.zeros(num_stab, dtype=np.float32)
    stab_y_all = np.zeros(num_stab, dtype=np.float32)
    stab_found = np.zeros(num_stab, dtype=bool)
    for i, sid in enumerate(stab_ids_arr):
        if not stab_found[sid]:
            stab_x_all[sid] = x_coords[i]
            stab_y_all[sid] = y_coords[i]
            stab_found[sid] = True

    # Convert coordinates to compact integer grid indices by mapping each unique
    # coordinate value to a consecutive index 0, 1, 2, ...
    # This is robust to any coordinate scale (e.g. AlphaQubit data uses spacing=4,
    # not 0.5, so the old coord_quant=0.5 approach inflated the grid by 8x).
    unique_x = np.unique(stab_x_all)
    unique_y = np.unique(stab_y_all)
    x_to_idx = {float(v): i for i, v in enumerate(unique_x)}
    y_to_idx = {float(v): i for i, v in enumerate(unique_y)}
    x_q = np.array([x_to_idx[float(stab_x_all[k])] for k in range(num_stab)], dtype=np.int32)
    y_q = np.array([y_to_idx[float(stab_y_all[k])] for k in range(num_stab)], dtype=np.int32)
    H = len(unique_x)
    W = len(unique_y)

    coord_to_index = {}
    index_to_coord = []
    for idx in range(num_stab):
        i, j = int(x_q[idx]), int(y_q[idx])
        coord_to_index[(i, j)] = idx
        index_to_coord.append((i, j))

    dilations = get_dilations(distance)
    channels_list = [model_cfg.conv_dim] * model_cfg.conv_layers

    blocks = nn.ModuleList(
        [
            ScatteringResidualConvBlock(
                d=distance,
                d_d=d_model,
                L_layers=model_cfg.conv_layers,
                channels_list=channels_list,
                coord_to_index=coord_to_index,
                index_to_coord=index_to_coord,
                dilation_list=dilations,
            )
            for _ in range(model_cfg.syndrome_layers)
        ]
    )
    return blocks, coord_to_index, index_to_coord


def build_model(
    layout: Dict[str, Any],
    model_cfg: ModelConfig,
    basis: str,
    use_full_bias: bool = True,
    max_rounds: Optional[int] = None,
    use_grad_checkpoint: bool = True,
) -> AlphaQubitModel:
    distance = layout["distance"]
    d_model = model_cfg.d_model

    conv_blocks, coord_to_index, index_to_coord = build_conv_blocks(
        distance, d_model, layout, model_cfg
    )

    if use_full_bias:
        if model_cfg.bias_mode == "split":
            bias_provider = AttentionBiasProviderSplit(
                db=model_cfg.bias_dim,
                max_dist=8,
                geom_resnet_layers=model_cfg.geom_resnet_layers,
                interaction_resnet_layers=model_cfg.interaction_resnet_layers,
                indicator_features=model_cfg.indicator_features,
            )
        else:  # "unified" — AlphaQubit-faithful default
            bias_provider = AttentionBiasProviderUnified(
                db=model_cfg.bias_dim,
                max_dist=8,
                num_residual_layers=model_cfg.bias_residual_layers,
                indicator_features=model_cfg.indicator_features,
            )
    else:
        bias_provider = ManhattanDistanceBias(db=model_cfg.bias_dim, max_dist=8)

    # num_cycles from layout["num_cycles"] is correct when layout is layout_ref
    # (the max-round folder). Circuits have rounds + 1 cycles (extra final-readout cycle),
    # so r25 gives num_cycles=26. max_rounds override kept for backward-compat callers.
    if max_rounds is not None:
        num_cycles = max_rounds + 1  # rounds + 1 final cycle
    else:
        num_cycles = layout["num_cycles"]

    model = AlphaQubitModel(
        num_stab=layout["num_stab"],
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
        index_to_coord=index_to_coord,
        basis=basis,
        use_grad_checkpoint=use_grad_checkpoint,
    )
    return model


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------


def pretrain_single(
    bases: List[str],
    distance: int,
    data_root: Path,
    checkpoint_dir: Path,
    rounds_list: List[int],
    train_cfg: Optional[TrainConfig] = None,
    model_cfg: Optional[ModelConfig] = None,
    use_full_bias: bool = True,
    use_wandb: bool = False,
    monolithic_file: Optional[Path] = None,
    chunk_size: int = 50_000,
) -> Dict[str, Any]:
    """Train one model for a given distance on all specified bases combined."""
    if train_cfg is None:
        train_cfg = pretrain_config(distance)
    if model_cfg is None:
        model_cfg = ModelConfig()

    bases_str = "+".join(b.upper() for b in sorted(bases))
    print(f"\n{'=' * 60}")
    print(f"PRETRAIN  bases={bases_str}  d={distance}  rounds={rounds_list}")
    print(f"{'=' * 60}")

    if monolithic_file is not None:
        from dataset_streaming import ChunkedHDF5Dataset, get_reference_layout

        # Each chunk holds BATCHES_PER_CHUNK batches.  When cur_bs grows the
        # chunk automatically grows so the ratio stays constant.
        BATCHES_PER_CHUNK = 200
        initial_chunk_size = BATCHES_PER_CHUNK * train_cfg.batch_init

        print(
            f"  Streaming mode: {monolithic_file}  "
            f"batches_per_chunk={BATCHES_PER_CHUNK}  "
            f"initial_chunk_size={initial_chunk_size}"
        )

        streaming_ds = ChunkedHDF5Dataset(
            monolithic_file,
            split="train",
            chunk_size=initial_chunk_size,
            distance=distance,
            seed=train_cfg.seed,
        )
        val_streaming_ds = ChunkedHDF5Dataset(
            monolithic_file,
            split="val",
            chunk_size=initial_chunk_size,
            distance=distance,
            seed=train_cfg.seed,
            shuffle=False,
        )
        layout = get_reference_layout(monolithic_file, distance)
        train_dataset = streaming_ds.load_chunk(0)
        val_dataset = val_streaming_ds.load_chunk(0)

        # --- Dynamic-size prefetch state ---
        # _ds_holder[0] is the active ChunkedHDF5Dataset (replaced on epoch wrap).
        # _next_off[0]  is the sample-index offset for the *next* chunk to load.
        # _seed_ctr[0]  is incremented on each epoch wrap.
        _ds_holder = [streaming_ds]
        _next_off = [initial_chunk_size]  # chunk-0 consumed [0 .. initial_chunk_size)
        _seed_ctr = [train_cfg.seed]
        _prefetch_buf: queue.Queue = queue.Queue(maxsize=3)

        def _load_and_enqueue(ds, offset, size):
            """Background worker: load one chunk and push it into _prefetch_buf."""
            try:
                chunk = ds.load_chunk(offset, chunk_size=size)
                _prefetch_buf.put(chunk)
            except Exception as exc:
                print(f"[prefetch] error at offset={offset}: {exc}")
                _prefetch_buf.put(None)  # signal failure so caller doesn't hang

        # Kick off prefetch for chunk 1 immediately.
        threading.Thread(
            target=_load_and_enqueue,
            args=(streaming_ds, initial_chunk_size, initial_chunk_size),
            daemon=True,
        ).start()
        _next_off[0] = 2 * initial_chunk_size  # chunk 2 will start here

        def get_next_chunk(cur_bs: int):
            """Return the next train chunk, adapting chunk size to cur_bs."""
            dynamic_size = BATCHES_PER_CHUNK * cur_bs

            # Block until the prefetched chunk is ready.
            chunk = _prefetch_buf.get(timeout=600)
            if chunk is None:
                chunk = None  # bubble up; train.py skips on None

            # Determine offset for the *next* prefetch.
            cur_ds = _ds_holder[0]
            next_off = _next_off[0]

            if next_off >= cur_ds.total_samples:
                # Epoch wrap — create a freshly shuffled dataset.
                _seed_ctr[0] += 1
                cur_ds = ChunkedHDF5Dataset(
                    monolithic_file,
                    split="train",
                    chunk_size=dynamic_size,
                    distance=distance,
                    seed=_seed_ctr[0],
                )
                _ds_holder[0] = cur_ds
                next_off = 0

            _next_off[0] = next_off + dynamic_size

            threading.Thread(
                target=_load_and_enqueue,
                args=(cur_ds, next_off, dynamic_size),
                daemon=True,
            ).start()

            return chunk
    else:
        train_dataset, val_dataset, layout = load_pretraining_datasets(
            data_root,
            bases,
            distance,
            rounds_list,
            n_train=16_000,
            n_val=4_000,
            seed=train_cfg.seed,
        )
        get_next_chunk = None

    # Build model with basis="x" as default; readout direction comes from batch basis_idx.
    # layout is layout_ref (max-round folder), so layout["num_cycles"] is already correct.
    model = build_model(layout, model_cfg, "x", use_full_bias=use_full_bias, use_grad_checkpoint=False)
    n_params = sum(p.numel() for p in model.parameters())
    device = torch.device(train_cfg.device)
    model.to(device)
    print(f"  Parameters: {n_params:,}   Device: {device}")
    print(f"  Train shots: {len(train_dataset)}  Val shots: {len(val_dataset)}")
    model.core = torch.compile(model.core, dynamic=True)
    model.bias_provider = torch.compile(model.bias_provider, dynamic=True)

    # No pos_weight: AlphaQubit uses plain BCE. The obs_rate imbalance (~0.36)
    # is mild enough that the model handles it without reweighting.

    run_name = f"pretrain_{bases_str.lower()}_d{distance}"
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
        get_next_train_chunk=get_next_chunk,
    )
    elapsed = time.time() - t0

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "layout": layout,
            "bases": bases,
            "distance": distance,
            "best_ler": best["ler"],
            "best_step": best["step"],
            "train_time": elapsed,
        },
        ckpt_path,
    )
    print(f"\n  Best LER: {best['ler']:.6f} at step {best['step']}")
    print(f"  Checkpoint: {ckpt_path}")

    if best.get("history"):
        save_history(best["history"], checkpoint_dir, run_name)

    return {
        "bases": bases,
        "distance": distance,
        "best_ler": best["ler"],
        "best_step": best["step"],
        "train_time": elapsed,
        "checkpoint_path": ckpt_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Trans7 pretraining on Google DEM-simulated data. "
        "Trains one model per distance on all bases combined.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--bases",
        nargs="+",
        choices=["x", "z"],
        default=["x", "z"],
        help="Bases to include in training (combined into one model per distance)",
    )
    parser.add_argument(
        "--distance",
        type=int,
        default=None,
        help="Single distance (overrides --distances)",
    )
    parser.add_argument(
        "--distances",
        type=int,
        nargs="+",
        default=[3],
        help="Distances to train (one model each)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        nargs="+",
        default=[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25],
        help="Round counts to include in pretraining data",
    )
    parser.add_argument("--data_dir", type=str, default="../../data/trans7_data")
    parser.add_argument(
        "--monolithic_file",
        type=str,
        default=None,
        help="Path to HDF5 file from data_random_sample.py. "
        "If set, uses streaming data loading instead of --data_dir.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=50_000,
        help="Samples per rolling window chunk (streaming mode only)",
    )
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/pretrain")
    parser.add_argument("--num_steps", type=int, default=1_000_000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--eval_every", type=int, default=15_000)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--no_full_bias", action="store_true")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    data_root = (script_dir / args.data_dir).resolve()
    checkpoint_dir = (script_dir / args.checkpoint_dir).resolve()

    repo_root = script_dir.parents[1]

    # If pretrain.h5 doesn't exist yet, build it now before training starts.
    h5_path = data_root / "pretrain.h5"
    if not h5_path.exists():
        import subprocess

        print(f"\n  pretrain.h5 not found — building from {data_root} ...")
        builder = repo_root / "data" / "data_random_sample.py"
        subprocess.run(
            [
                sys.executable,
                str(builder),
                "--data_dir",
                str(data_root),
                "--output",
                str(h5_path),
                "--seed",
                "42",
            ],
            check=True,
        )
        print()

    distances = [args.distance] if args.distance else args.distances

    results = []
    for distance in distances:
        cfg = pretrain_config(distance)
        cfg.num_steps = args.num_steps
        cfg.batch_init = args.batch_size
        cfg.eval_every = args.eval_every
        if args.lr is not None:
            cfg.lr = args.lr

        model_cfg = ModelConfig()
        model_cfg.d_model = args.d_model

        mono = (
            (script_dir / args.monolithic_file).resolve()
            if args.monolithic_file
            else h5_path
        )

        result = pretrain_single(
            bases=args.bases,
            distance=distance,
            data_root=data_root,
            checkpoint_dir=checkpoint_dir,
            rounds_list=args.rounds,
            train_cfg=cfg,
            model_cfg=model_cfg,
            use_full_bias=not args.no_full_bias,
            use_wandb=args.use_wandb,
            monolithic_file=mono,
            chunk_size=args.chunk_size,
        )
        results.append(result)

    print(f"\n{'=' * 60}\nPRETRAINING SUMMARY\n{'=' * 60}")
    for r in results:
        bases_str = "+".join(b.upper() for b in sorted(r["bases"]))
        print(
            f"  bases={bases_str} d={r['distance']}  "
            f"LER={r['best_ler']:.6f}  step={r['best_step']}  "
            f"time={r['train_time']:.0f}s"
        )

    idx_path = checkpoint_dir / "pretrain_index.json"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with open(idx_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Index saved to {idx_path}")


if __name__ == "__main__":
    main()
