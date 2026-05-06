#!/usr/bin/env python3
"""
eval_lambda.py - Evaluate trans6 model and compute Lambda (error suppression factor).

Generates fresh evaluation data at multiple round counts per distance,
runs model inference, computes per-round error rate epsilon, and
calculates Lambda across distances following AlphaQubit's methodology.

Optionally runs MWPM on the same circuits for side-by-side comparison.

Usage:
    python eval_lambda.py --checkpoint checkpoints/z_d3_r25.pth --distances 3 5 --basis z
    python eval_lambda.py --checkpoint checkpoints/z_d3_r25.pth \\
        --distances 3 5 --eval_rounds 6 10 15 20 25 --shots 100000
    python eval_lambda.py --checkpoint checkpoints/z_d3_r25.pth --distances 3 5 --mwpm
"""

import argparse
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

import stim

from model import AlphaQubitLikeModel, ScatteringResidualConvBlock
from dataset import SyndromeDataset, make_loader
from parameter import ModelConfigScaling, ScalingConfig
from layout import build_layout_from_circuit, save_layout_json
from gen_basis_data import generate_basis_dataset


def ler_to_epsilon(ler: float, rounds: int) -> float:
    """Convert total LER E(n) to per-round error rate epsilon.

    From AlphaQubit paper Eq. 4:
        epsilon = (1/2) * (1 - (1 - 2*E(n))^(1/n))
    """
    if ler <= 0.0:
        return 0.0
    if ler >= 0.5:
        return 0.5
    return 0.5 * (1.0 - (1.0 - 2.0 * ler) ** (1.0 / rounds))


def compute_lambda(eps_small_d: float, eps_large_d: float) -> float:
    """Compute Lambda = epsilon(d_small) / epsilon(d_large)."""
    if eps_large_d <= 0:
        return float("inf")
    return eps_small_d / eps_large_d


def extend_cycle_embedding(model: AlphaQubitLikeModel, new_num_cycles: int):
    """Extend cycle embedding if eval requires more cycles than training had."""
    old_emb = model.embed.cycle_emb
    old_num = old_emb.num_embeddings
    if new_num_cycles <= old_num:
        return

    device = old_emb.weight.device
    new_emb = nn.Embedding(new_num_cycles, old_emb.embedding_dim).to(device)
    with torch.no_grad():
        new_emb.weight[:old_num] = old_emb.weight
        # Extrapolate: copy last embedding for new positions
        new_emb.weight[old_num:] = old_emb.weight[-1]
    model.embed.cycle_emb = new_emb

    # Also extend final_embed if it exists
    if hasattr(model, "final_embed") and model.final_embed is not None:
        old_final = model.final_embed.cycle_emb
        old_final_num = old_final.num_embeddings
        if new_num_cycles > old_final_num:
            new_final = nn.Embedding(new_num_cycles, old_final.embedding_dim).to(device)
            with torch.no_grad():
                new_final.weight[:old_final_num] = old_final.weight
                new_final.weight[old_final_num:] = old_final.weight[-1]
            model.final_embed.cycle_emb = new_final

    print(f"  Extended cycle embedding: {old_num} -> {new_num_cycles}")


def rebuild_conv_blocks(distance: int, d_model: int, layout: dict,
                        model_cfg: ModelConfigScaling, n_layers: int):
    """Build conv blocks from layout for a specific distance."""
    from run_train import build_conv_block
    return nn.ModuleList([
        build_conv_block(distance, d_model, layout, model_cfg)
        for _ in range(n_layers)
    ])


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device,
                               layout: dict, basis: str) -> AlphaQubitLikeModel:
    """Load a trained model from checkpoint, adapting to eval layout if needed."""
    from run_train import build_model
    from utils import AttentionBiasProvider

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_cfg_dict = ckpt.get("model_cfg", {})
    model_cfg = ModelConfigScaling(**model_cfg_dict)

    # Build model with eval layout
    train_cfg = ScalingConfig()
    model = build_model(layout, model_cfg, train_cfg, use_full_bias=True, basis=basis)

    # Load weights (allow size mismatches for embeddings)
    state_dict = ckpt.get("model_state_dict", ckpt)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        # Handle embedding size mismatch
        model.load_state_dict(state_dict, strict=False)
        print("  Warning: loaded with strict=False (embedding size mismatch)")

    model = model.to(device)
    model.eval()
    return model, model_cfg


def evaluate_at_rounds(model, layout, basis, distance, rounds, p, shots,
                       seed, device, input_mode="hard") -> dict:
    """Generate eval data at specific rounds and compute LER."""
    # Generate fresh data
    circ, det_hard, meas_hard, obs = generate_basis_dataset(
        basis=basis, distance=distance, rounds=rounds,
        p=p, shots=shots, seed=seed,
    )

    eval_layout = build_layout_from_circuit(circ)
    eval_layout["distance"] = distance

    # Extend cycle embedding if needed
    eval_num_cycles = eval_layout["num_cycles"]
    extend_cycle_embedding(model, eval_num_cycles)

    # Create temporary layout file for SyndromeDataset
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(eval_layout, f)
        tmp_layout_path = f.name

    try:
        dataset = SyndromeDataset(
            samples=det_hard,
            labels=obs,
            layout_json_path=tmp_layout_path,
            input_mode=input_mode,
            measurements=meas_hard,
        )

        loader = make_loader(dataset, batch_size=256, shuffle=False)

        all_logits = []
        all_labels = []

        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                         for k, v in batch.items()}
                out = model(batch)

                logits = out.get("logical_logits", out.get("logits", None))
                if logits is None:
                    logits = out.get("logical_logits_x", out.get("logical_logits_z"))
                logits = logits.detach().cpu().view(-1)
                all_logits.append(logits)

                labels = batch.get("logical_labels", batch.get("label", None))
                labels = labels.detach().cpu().view(-1)
                all_labels.append(labels)

        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)

        preds = (torch.sigmoid(all_logits) > 0.5).float()
        num_errors = int((preds != all_labels).sum())
        ler = num_errors / len(all_labels)
        epsilon = ler_to_epsilon(ler, rounds)

    finally:
        os.unlink(tmp_layout_path)

    return {
        "distance": distance,
        "rounds": rounds,
        "ler": ler,
        "epsilon": epsilon,
        "num_errors": num_errors,
        "shots": len(all_labels),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trans6 model Lambda (error suppression factor)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--distances", type=int, nargs="+", default=[3, 5],
                        help="Code distances to evaluate")
    parser.add_argument("--eval_rounds", type=int, nargs="+", default=None,
                        help="Explicit round counts (same for all distances). "
                             "Default: distance * [2, 3, 4, 5]")
    parser.add_argument("--rounds_multipliers", type=int, nargs="+", default=[2, 3, 4, 5],
                        help="Multipliers for rounds = distance * multiplier (if --eval_rounds not set)")
    parser.add_argument("--basis", type=str, default="z", choices=["z", "x"],
                        help="Measurement basis")
    parser.add_argument("--p", type=float, default=0.005,
                        help="Physical error rate")
    parser.add_argument("--shots", type=int, default=100000,
                        help="Shots per evaluation point")
    parser.add_argument("--seed", type=int, default=77777,
                        help="Base seed for eval data generation")
    parser.add_argument("--mwpm", action="store_true",
                        help="Also run MWPM and show comparison")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Trans6 Lambda Evaluation")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Basis: {args.basis.upper()}")
    print(f"  Physical error rate: {args.p}")
    print(f"  Shots: {args.shots:,}")
    print(f"  Device: {device}")
    print()

    # Determine round counts per distance
    if args.eval_rounds is not None:
        rounds_per_distance = {d: args.eval_rounds for d in args.distances}
    else:
        rounds_per_distance = {
            d: [d * m for m in args.rounds_multipliers]
            for d in args.distances
        }

    # For model loading, use the first distance's first round count to build initial layout
    first_d = sorted(args.distances)[0]
    first_r = sorted(rounds_per_distance[first_d])[0]

    circ_init = stim.Circuit.generated(
        f"surface_code:rotated_memory_{args.basis}",
        distance=first_d, rounds=first_r,
        after_clifford_depolarization=args.p,
        before_round_data_depolarization=args.p,
        before_measure_flip_probability=args.p,
        after_reset_flip_probability=args.p,
    )
    init_layout = build_layout_from_circuit(circ_init)
    init_layout["distance"] = first_d

    model, model_cfg = load_model_from_checkpoint(
        args.checkpoint, device, init_layout, args.basis,
    )
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Evaluate trans6
    trans6_results = []
    trans6_epsilon_by_distance = {}

    for d in sorted(args.distances):
        rounds_list = sorted(rounds_per_distance[d])
        epsilons = []

        for r in rounds_list:
            print(f"  [Trans6] d={d}, r={r}...", end=" ", flush=True)
            result = evaluate_at_rounds(
                model, init_layout, args.basis, d, r, args.p,
                args.shots, args.seed + d * 100 + r, device,
            )
            result["decoder"] = "Trans6"
            trans6_results.append(result)
            epsilons.append(result["epsilon"])
            print(f"LER={result['ler']:.6f}  epsilon={result['epsilon']:.6f}")

        trans6_epsilon_by_distance[d] = float(np.mean(epsilons))

    # Evaluate MWPM if requested
    mwpm_results = []
    mwpm_epsilon_by_distance = {}

    if args.mwpm:
        from eval_mwpm import mwpm_decode
        print()
        for d in sorted(args.distances):
            rounds_list = sorted(rounds_per_distance[d])
            epsilons = []

            for r in rounds_list:
                print(f"  [MWPM]   d={d}, r={r}...", end=" ", flush=True)
                result = mwpm_decode(
                    basis=args.basis, distance=d, rounds=r,
                    p=args.p, shots=args.shots,
                    seed=args.seed + d * 100 + r,
                )
                result["decoder"] = "MWPM"
                mwpm_results.append(result)
                epsilons.append(result["epsilon"])
                print(f"LER={result['ler']:.6f}  epsilon={result['epsilon']:.6f}")

            mwpm_epsilon_by_distance[d] = float(np.mean(epsilons))

    # Print results
    print(f"\n{'='*90}")
    print(f"{'Decoder':<10} {'Dist':>6} {'Rounds':>8} {'LER':>12} {'epsilon':>12} {'Errors':>10}")
    print(f"{'-'*90}")
    for r in trans6_results:
        print(f"{'Trans6':<10} {r['distance']:>6} {r['rounds']:>8} "
              f"{r['ler']:>12.6f} {r['epsilon']:>12.6f} {r['num_errors']:>10,}")
    for r in mwpm_results:
        print(f"{'MWPM':<10} {r['distance']:>6} {r['rounds']:>8} "
              f"{r['ler']:>12.6f} {r['epsilon']:>12.6f} {r['num_errors']:>10,}")

    # Compute Lambda
    sorted_distances = sorted(args.distances)
    if len(sorted_distances) >= 2:
        print(f"\n{'='*90}")
        print("Lambda (error suppression factors):")
        print(f"{'='*90}")

        d_min = sorted_distances[0]
        for d_large in sorted_distances[1:]:
            lam_trans6 = compute_lambda(
                trans6_epsilon_by_distance[d_min],
                trans6_epsilon_by_distance[d_large],
            )
            line = (f"  Trans6  Lambda_{d_min}/{d_large} = {lam_trans6:.4f}  "
                    f"(eps_{d_min}={trans6_epsilon_by_distance[d_min]:.6f}, "
                    f"eps_{d_large}={trans6_epsilon_by_distance[d_large]:.6f})")
            print(line)

            if args.mwpm and d_large in mwpm_epsilon_by_distance:
                lam_mwpm = compute_lambda(
                    mwpm_epsilon_by_distance[d_min],
                    mwpm_epsilon_by_distance[d_large],
                )
                line = (f"  MWPM    Lambda_{d_min}/{d_large} = {lam_mwpm:.4f}  "
                        f"(eps_{d_min}={mwpm_epsilon_by_distance[d_min]:.6f}, "
                        f"eps_{d_large}={mwpm_epsilon_by_distance[d_large]:.6f})")
                print(line)

    # Save results
    output_path = args.output
    if output_path is None:
        script_dir = Path(__file__).parent
        output_path = str(script_dir / "lambda_results.json")

    output = {
        "checkpoint": args.checkpoint,
        "basis": args.basis,
        "p": args.p,
        "shots": args.shots,
        "trans6_results": trans6_results,
        "trans6_epsilon_by_distance": {str(k): v for k, v in trans6_epsilon_by_distance.items()},
    }
    if args.mwpm:
        output["mwpm_results"] = mwpm_results
        output["mwpm_epsilon_by_distance"] = {str(k): v for k, v in mwpm_epsilon_by_distance.items()}

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
