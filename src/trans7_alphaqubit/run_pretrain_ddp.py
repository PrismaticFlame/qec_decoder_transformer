#!/usr/bin/env python3
"""
run_pretrain_ddp.py - Multi-GPU pretraining via DistributedDataParallel.

Launch with torchrun:
    torchrun --nproc_per_node=2 run_pretrain_ddp.py --distance 3 --num_steps 500000
    torchrun --nproc_per_node=4 run_pretrain_ddp.py --distances 3 5

Same CLI arguments as run_pretrain.py.  Only rank 0 logs, saves checkpoints,
and runs validation.  All ranks participate in forward/backward.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from hyperparameters import ModelConfig, TrainConfig, pretrain_config
from run_pretrain import (
    build_model,
    find_folders,
    load_pretraining_datasets,
)
from torch.nn.parallel import DistributedDataParallel as DDP

from train import save_history, train


def setup_distributed():
    """Initialize the distributed process group and return rank/world_size."""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_distributed():
    dist.destroy_process_group()


def pretrain_single_ddp(
    bases: List[str],
    distance: int,
    data_root: Path,
    checkpoint_dir: Path,
    rounds_list: List[int],
    rank: int,
    world_size: int,
    local_rank: int,
    train_cfg: Optional[TrainConfig] = None,
    model_cfg: Optional[ModelConfig] = None,
    use_full_bias: bool = True,
    use_wandb: bool = False,
    monolithic_file: Optional[Path] = None,
) -> Dict[str, Any]:
    """Train one model for a given distance using DDP across multiple GPUs."""
    if train_cfg is None:
        train_cfg = pretrain_config(distance)
    if model_cfg is None:
        model_cfg = ModelConfig()

    is_main = rank == 0
    bases_str = "+".join(b.upper() for b in sorted(bases))

    if is_main:
        print(f"\n{'=' * 60}")
        print(
            f"PRETRAIN (DDP, {world_size} GPUs)  bases={bases_str}  d={distance}  rounds={rounds_list}"
        )
        print(f"{'=' * 60}")

    # Force device to this rank's GPU
    train_cfg.device = f"cuda:{local_rank}"

    # Load data — streaming or folder-based
    if monolithic_file is not None:
        from dataset_streaming import ChunkedHDF5Dataset, get_reference_layout

        initial_chunk_size = 200 * train_cfg.batch_init
        if is_main:
            print(
                f"  Streaming mode: {monolithic_file}  chunk_size={initial_chunk_size}"
            )

        streaming_ds = ChunkedHDF5Dataset(
            monolithic_file,
            chunk_size=initial_chunk_size,
            distance=distance,
            seed=train_cfg.seed,
        )
        layout = get_reference_layout(monolithic_file, distance)
        train_dataset, val_dataset = streaming_ds.load_chunk_split(
            0,
            val_fraction=0.2,
            seed=train_cfg.seed,
        )
        # Streaming chunk prefetch not implemented for DDP yet — use simple reload
        get_next_chunk = None
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

    # Build model and wrap in DDP
    model = build_model(layout, model_cfg, "x", use_full_bias=use_full_bias)
    device = torch.device(train_cfg.device)
    model.to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    n_params = sum(p.numel() for p in model.parameters())
    if is_main:
        print(
            f"  Parameters: {n_params:,}   Device: {device}   World size: {world_size}"
        )
        print(f"  Train shots: {len(train_dataset)}  Val shots: {len(val_dataset)}")

    run_name = f"pretrain_{bases_str.lower()}_d{distance}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = str(checkpoint_dir / f"{run_name}.pth")

    # Synchronize before training
    dist.barrier()

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
        rank=rank,
        world_size=world_size,
    )
    elapsed = time.time() - t0

    # Only rank 0 saves final checkpoint
    if is_main:
        # Save underlying model state (unwrap DDP)
        raw_model = model.module
        torch.save(
            {
                "model_state_dict": raw_model.state_dict(),
                "layout": layout,
                "bases": bases,
                "distance": distance,
                "best_ler": best["ler"],
                "best_step": best["step"],
                "train_time": elapsed,
                "world_size": world_size,
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
        description="Trans7 DDP pretraining (multi-GPU). "
        "Launch with: torchrun --nproc_per_node=N run_pretrain_ddp.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--bases", nargs="+", choices=["x", "z"], default=["x", "z"])
    parser.add_argument("--distance", type=int, default=None)
    parser.add_argument("--distances", type=int, nargs="+", default=[3, 5])
    parser.add_argument(
        "--rounds",
        type=int,
        nargs="+",
        default=[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25],
    )
    parser.add_argument("--data_dir", type=str, default="../../data/trans7_data")
    parser.add_argument("--monolithic_file", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/pretrain")
    parser.add_argument("--num_steps", type=int, default=2_000_000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--eval_every", type=int, default=5_000)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--no_full_bias", action="store_true")
    args = parser.parse_args()

    rank, world_size, local_rank = setup_distributed()
    is_main = rank == 0

    script_dir = Path(__file__).parent
    data_root = (script_dir / args.data_dir).resolve()
    checkpoint_dir = (script_dir / args.checkpoint_dir).resolve()

    # Data preparation — only rank 0
    if is_main:
        from move_surface_code_dirs import ensure_surface_code_data

        repo_root = script_dir.parents[1]
        if not ensure_surface_code_data(repo_root / "data", data_root):
            print(
                f"ERROR: No surface_code_b* directories found under {repo_root / 'data'}",
                file=sys.stderr,
            )
            dist.destroy_process_group()
            sys.exit(1)

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

    # Wait for rank 0 to finish data prep
    dist.barrier()

    distances = [args.distance] if args.distance else args.distances
    h5_path = data_root / "pretrain.h5"

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

        result = pretrain_single_ddp(
            bases=args.bases,
            distance=distance,
            data_root=data_root,
            checkpoint_dir=checkpoint_dir,
            rounds_list=args.rounds,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            train_cfg=cfg,
            model_cfg=model_cfg,
            use_full_bias=not args.no_full_bias,
            use_wandb=args.use_wandb,
            monolithic_file=mono,
        )
        results.append(result)

    if is_main:
        print(f"\n{'=' * 60}\nPRETRAINING SUMMARY (DDP, {world_size} GPUs)\n{'=' * 60}")
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

    cleanup_distributed()


if __name__ == "__main__":
    main()
