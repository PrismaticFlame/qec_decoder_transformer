# eval.py
# Independent evaluation module for QEC transformer (no optimizer interaction)
from __future__ import annotations

import math
from typing import Dict, Any, Optional, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from .ema import EMA
from .utils import fit_line_with_stats, gate_fit_ok


# -------------------------
# LER computation from logits (safe)
# -------------------------
@torch.no_grad()
def compute_ler_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute Logical Error Rate (LER) from logits and labels.
    
    Args:
        logits: (N,) or (N, 1) logits
        labels: (N,) or (N, 1) in {0, 1}
    
    Returns:
        LER: float in [0, 1]
    """
    logits = logits.view(-1)
    labels = labels.view(-1).float()
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    return float((preds != labels).float().mean().item())


def _extract_logits_from_model_output(out) -> torch.Tensor:
    """
    Extract logits from model output.
    
    Model output can be:
      - dict with 'logical_logits' (preferred)
      - dict with 'logits' or 'logit' (legacy)
      - tensor logits directly
    
    For X/Z separate mode, returns combined logits or X logits.
    """
    if isinstance(out, dict):
        # Try X/Z separate mode first
        if "logical_logits_x" in out and "logical_logits_z" in out:
            # Combine X and Z logits (average or sum, depending on your preference)
            logits_x = out["logical_logits_x"].view(-1)
            logits_z = out["logical_logits_z"].view(-1)
            # Average for combined LER
            return (logits_x + logits_z) / 2.0
        
        # Fallback to single logits
        if "logical_logits" in out:
            return out["logical_logits"]
        if "logit" in out:
            return out["logit"]
        if "logits" in out:
            return out["logits"]
        raise KeyError("model output dict but no 'logical_logits'/'logit'/'logits' key found.")
    
    if torch.is_tensor(out):
        return out
    
    raise TypeError("model output must be a dict or a Tensor logits.")


# -------------------------
# Cycle evaluation helpers
# -------------------------
@torch.no_grad()
def compute_cycle_ler(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    cycles: List[int],
    *,
    use_ema: bool = True,
    ema: Optional[EMA] = None,
    return_xz: bool = False,
) -> Dict[int, float]:
    """
    Compute LER for specified cycles.
    
    Args:
        model: Model to evaluate
        val_loader: Validation data loader
        device: Device to run on
        cycles: List of cycle numbers to evaluate (for compatibility, actual LER is computed on all data)
        use_ema: Whether to use EMA model (if ema is provided)
        ema: EMA object (optional)
        return_xz: If True, return separate X and Z LERs
    
    Returns:
        Dictionary mapping cycle -> LER (or tuple of (LER, LER_X, LER_Z) if return_xz=True)
    """
    model.eval()
    
    # Apply EMA if requested
    if use_ema and ema is not None:
        ema.apply_to(model)
    
    try:
        all_logits = []
        all_labels = []
        all_logits_x = []
        all_logits_z = []
        all_labels_x = []
        all_labels_z = []
        
        for batch in val_loader:
            if isinstance(batch, dict):
                batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
                # Support both legacy and new label formats
                labels = batch.get("logical_labels", batch.get("label", None))
                labels_x = batch.get("label_x", None)
                labels_z = batch.get("label_z", None)
            else:
                batch = [v.to(device, non_blocking=True) if torch.is_tensor(v) else v for v in batch]
                labels = None
                labels_x = None
                labels_z = None
            
            out = model(batch)
            
            # Extract logits
            logits = _extract_logits_from_model_output(out)
            all_logits.append(logits.detach())
            
            # Extract X/Z logits if available
            if isinstance(out, dict):
                if "logical_logits_x" in out:
                    all_logits_x.append(out["logical_logits_x"].view(-1).detach())
                if "logical_logits_z" in out:
                    all_logits_z.append(out["logical_logits_z"].view(-1).detach())
            
            # Get labels
            if labels is None:
                if isinstance(out, dict) and ("logical_labels" in out or "label" in out):
                    labels = out.get("logical_labels", out.get("label"))
                else:
                    raise KeyError("Cannot find labels. Expect batch['logical_labels'] (or legacy batch['label']).")
            
            all_labels.append(labels.view(-1).detach())
            
            if labels_x is not None:
                all_labels_x.append(labels_x.view(-1).detach())
            if labels_z is not None:
                all_labels_z.append(labels_z.view(-1).detach())
        
        # Compute combined LER
        logits_cat = torch.cat([t.view(-1) for t in all_logits], dim=0)
        labels_cat = torch.cat([t.view(-1) for t in all_labels], dim=0)
        ler = compute_ler_from_logits(logits_cat, labels_cat)
        
        # Compute X/Z LERs if available
        ler_x = None
        ler_z = None
        if return_xz and len(all_logits_x) > 0 and len(all_logits_z) > 0:
            logits_x_cat = torch.cat(all_logits_x, dim=0)
            logits_z_cat = torch.cat(all_logits_z, dim=0)
            if len(all_labels_x) > 0 and len(all_labels_z) > 0:
                labels_x_cat = torch.cat(all_labels_x, dim=0)
                labels_z_cat = torch.cat(all_labels_z, dim=0)
                ler_x = compute_ler_from_logits(logits_x_cat, labels_x_cat)
                ler_z = compute_ler_from_logits(logits_z_cat, labels_z_cat)
        
        # Return same LER for all cycles (for compatibility)
        result = {int(c): float(ler) for c in cycles}
        
        if return_xz and ler_x is not None and ler_z is not None:
            # Store X/Z LERs in result
            for c in cycles:
                result[f"{c}_x"] = float(ler_x)
                result[f"{c}_z"] = float(ler_z)
        
        return result
    
    finally:
        # Restore original model if EMA was applied
        if use_ema and ema is not None:
            ema.restore(model)


@torch.no_grad()
def evaluate_ler_with_fit(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    ema: Optional[EMA],
    cycles_for_fit: List[int],
    min_r2: float,
    min_intercept: float,
    *,
    use_ema: bool = True,
    return_xz: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate LER with linear fit (Sycamore-style evaluation).
    
    Args:
        model: Model to evaluate
        val_loader: Validation data loader
        device: Device to run on
        ema: EMA object (optional)
        cycles_for_fit: List of cycles to use for fitting
        min_r2: Minimum R² for fit to be considered valid
        min_intercept: Minimum intercept threshold
        use_ema: Whether to use EMA model (if ema is provided)
        return_xz: If True, also compute and return X/Z LERs
    
    Returns:
        Dictionary with metrics:
            - dev/ler_pred25: Predicted LER at cycle 25
            - dev/fit_r2: R² of linear fit
            - dev/fit_slope: Slope of linear fit
            - dev/fit_intercept: Intercept of linear fit
            - dev/fit_sigma_intercept: Standard error of intercept
            - dev/fit_intercept_thresh: Intercept threshold
            - dev/fit_ok: Whether fit passes quality gates
            - dev/ler_cycle_{c}: LER for each cycle c
            - (if return_xz) dev/ler_x, dev/ler_z, dev/ler_x_cycle_{c}, dev/ler_z_cycle_{c}
    """
    model.eval()
    
    if use_ema and ema is not None:
        ema.apply_to(model)
    
    try:
        ler_by_cycle = compute_cycle_ler(
            model, val_loader, device, cycles_for_fit,
            use_ema=False,  # Already applied
            ema=None,
            return_xz=return_xz,
        )
        
        # Extract combined LERs (remove X/Z keys if present)
        ler_dict = {k: v for k, v in ler_by_cycle.items() if not str(k).endswith(('_x', '_z'))}
        if not ler_dict:
            ler_dict = ler_by_cycle
        
        x = np.array(cycles_for_fit, dtype=np.float64)
        y = np.array([ler_dict.get(int(c), ler_dict.get(c, 0.0)) for c in cycles_for_fit], dtype=np.float64)
        
        stats = fit_line_with_stats(x, y)
        ok, intercept_thresh = gate_fit_ok(stats, min_r2=min_r2, min_intercept=min_intercept)
        
        pred_25 = (
            stats["slope"] * 25.0 + stats["intercept"]
            if (not math.isnan(stats["slope"]) and not math.isnan(stats["intercept"]))
            else float("nan")
        )
        
        out = {
            "dev/ler_pred25": float(pred_25),
            "dev/fit_r2": float(stats["r2"]),
            "dev/fit_slope": float(stats["slope"]),
            "dev/fit_intercept": float(stats["intercept"]),
            "dev/fit_sigma_intercept": float(stats["sigma_intercept"]),
            "dev/fit_intercept_thresh": float(intercept_thresh),
            "dev/fit_ok": float(1.0 if ok else 0.0),
        }
        
        # Add per-cycle LERs
        for c in cycles_for_fit:
            key = int(c)
            if key in ler_dict:
                out[f"dev/ler_cycle_{key}"] = float(ler_dict[key])
        
        # Add X/Z LERs if requested
        if return_xz:
            ler_x = ler_by_cycle.get(f"{cycles_for_fit[0]}_x", None)
            ler_z = ler_by_cycle.get(f"{cycles_for_fit[0]}_z", None)
            if ler_x is not None and ler_z is not None:
                out["dev/ler_x"] = float(ler_x)
                out["dev/ler_z"] = float(ler_z)
                for c in cycles_for_fit:
                    key = int(c)
                    ler_x_c = ler_by_cycle.get(f"{key}_x", None)
                    ler_z_c = ler_by_cycle.get(f"{key}_z", None)
                    if ler_x_c is not None:
                        out[f"dev/ler_x_cycle_{key}"] = float(ler_x_c)
                    if ler_z_c is not None:
                        out[f"dev/ler_z_cycle_{key}"] = float(ler_z_c)
        
        return out
    
    finally:
        if use_ema and ema is not None:
            ema.restore(model)


# -------------------------
# Main evaluation function
# -------------------------
@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    *,
    ema: Optional[EMA] = None,
    use_ema: bool = True,
    cycles: Optional[List[int]] = None,
    eval_mode: str = "sycamore",
    min_r2: float = 0.9,
    min_intercept: float = -0.02,
    return_xz: bool = False,
) -> Dict[str, Any]:
    """
    Main evaluation function.
    
    Args:
        model: Model to evaluate
        val_loader: Validation data loader
        device: Device to run on
        ema: EMA object (optional)
        use_ema: Whether to use EMA model (if ema is provided)
        cycles: List of cycles to evaluate (default: [3, 5, 7, 9, 11, 13, 15, 25])
        eval_mode: Evaluation mode - "sycamore" (with fit) or "simple" (cycle 25 only)
        min_r2: Minimum R² for fit (sycamore mode)
        min_intercept: Minimum intercept threshold (sycamore mode)
        return_xz: If True, also compute and return X/Z LERs
    
    Returns:
        Dictionary with evaluation metrics
    """
    if cycles is None:
        cycles = [3, 5, 7, 9, 11, 13, 15, 25]
    
    eval_mode = eval_mode.lower()
    
    if eval_mode == "sycamore":
        return evaluate_ler_with_fit(
            model=model,
            val_loader=val_loader,
            device=device,
            ema=ema,
            cycles_for_fit=cycles,
            min_r2=min_r2,
            min_intercept=min_intercept,
            use_ema=use_ema,
            return_xz=return_xz,
        )
    else:
        # Simple mode: just compute LER for cycle 25
        ler_by_cycle = compute_cycle_ler(
            model=model,
            val_loader=val_loader,
            device=device,
            cycles=[25],
            use_ema=use_ema,
            ema=ema,
            return_xz=return_xz,
        )
        
        result = {
            "dev/ler_cycle_25": float(ler_by_cycle[25]),
            "dev/fit_ok": 1.0,
        }
        
        if return_xz:
            ler_x = ler_by_cycle.get("25_x", None)
            ler_z = ler_by_cycle.get("25_z", None)
            if ler_x is not None:
                result["dev/ler_x"] = float(ler_x)
            if ler_z is not None:
                result["dev/ler_z"] = float(ler_z)
        
        return result


# -------------------------
# Test set evaluation
# -------------------------
@torch.no_grad()
def evaluate_test_set(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    *,
    ema: Optional[EMA] = None,
    use_ema: bool = True,
    cycles: Optional[List[int]] = None,
    return_xz: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate on test set (same interface as evaluate_model).
    
    Uses "test/" prefix instead of "dev/" for metrics.
    """
    if cycles is None:
        cycles = [3, 5, 7, 9, 11, 13, 15, 25]
    
    # Use sycamore mode for test evaluation
    result = evaluate_ler_with_fit(
        model=model,
        val_loader=test_loader,
        device=device,
        ema=ema,
        cycles_for_fit=cycles,
        min_r2=0.9,
        min_intercept=-0.02,
        use_ema=use_ema,
        return_xz=return_xz,
    )
    
    # Rename keys from "dev/" to "test/"
    test_result = {}
    for k, v in result.items():
        new_k = k.replace("dev/", "test/")
        test_result[new_k] = v
    
    return test_result
