# loss.py
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Any


class QECMultiTaskLoss(nn.Module):
    """
    Multi-task loss:
      (1) logical flip prediction (binary)
      (2) next-stabilizer prediction (binary per token) [optional]

    Additions:
      - supports masking tokens (padding/leakage/terminal round)
      - optional pos_weight for class imbalance
      - optional scheduling (anneal) for next-stabilizer task weight
    """

    def __init__(
        self,
        next_stab_weight: float = 0.1,                  # base weight for stabilizer head
        logical_pos_weight: Optional[float] = None,     # e.g., 50~500 depending on rarity
        stab_pos_weight: Optional[float] = None,        # optional, often None
        reduction: str = "mean",                        # "mean" | "sum" | "none"
        # schedule config for next_stab_weight 
        next_stab_weight_min: float = 0.0,              # final/min stabilizer weight after anneal
        next_stab_schedule: str = "none",               # "none" | "linear" | "cosine" | "piecewise"
        warmup_ratio: float = 0.3,                      # for "piecewise": keep w0 before this fraction
        decay_ratio: float = 0.5,                       # for "piecewise": decay duration fraction after warmup
    ):
        super().__init__()
        self.next_stab_weight = float(next_stab_weight)
        self.next_stab_weight_min = float(next_stab_weight_min)
        self.next_stab_schedule = str(next_stab_schedule).lower()
        self.warmup_ratio = float(warmup_ratio)
        self.decay_ratio = float(decay_ratio)

        self.reduction = reduction

        # store as buffers so they move with .to(device)但不會被optimizer更新, 如果要被optimizer更新會存在nn.Parameter
        self.register_buffer(
            "logical_pos_weight",
            torch.tensor([logical_pos_weight], dtype=torch.float32) if logical_pos_weight is not None else None
        )
        self.register_buffer(
            "stab_pos_weight",
            torch.tensor([stab_pos_weight], dtype=torch.float32) if stab_pos_weight is not None else None
        )

    def _bce_logits(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        pos_weight: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        logits/targets: same shape
        mask: broadcastable bool/float mask where 1=keep, 0=ignore
        """
        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=pos_weight,
            reduction="none",
        )

        if mask is not None: #leakage=0, padding=0, last round=1
            mask_f = mask.float() if mask.dtype != torch.bool else mask.float()
            loss = loss * mask_f

            # masked mean (stable)
            denom = mask_f.sum().clamp_min(1.0) #mask_f.sum()：計算有效位元總數, .clamp_min(1.0)：如果元素的值小於 1.0，則該元素的值會被替換為 1.0。 防止除以零 (Zero Division)
            return loss.sum() / denom

        # default reduction 沒有 mask 時的標準 loss reduction 行為
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss  # "none"

    def _scheduled_next_stab_weight(
        self,
        step: Optional[int],
        total_steps: Optional[int],
    ) -> float:
        """
        AlphaQubit-style auxiliary loss weight schedule.

        Behavior:
        - schedule == "alphaqubit":
            * warmup phase: keep w0
            * after warmup: cosine decay from w0 to w_min
        - otherwise: fallback to constant w0
        """
        w0 = float(self.next_stab_weight)
        w_min = float(self.next_stab_weight_min)

        # safety / fallback
        if self.next_stab_schedule != "alphaqubit":
            return w0
        if step is None or total_steps is None or total_steps <= 0:
            return w0

        # progress p in [0,1]
        p = float(step) / float(total_steps)
        p = max(0.0, min(1.0, p))

        # warmup ratio in [0,1]
        warm = max(0.0, min(1.0, self.warmup_ratio))

        # warmup: keep full auxiliary weight
        if p <= warm:
            return w0

        # rescale progress after warmup → q in [0,1]
        q = (p - warm) / max(1e-8, (1.0 - warm))
        q = max(0.0, min(1.0, q))

        # cosine annealing: w0 -> w_min
        return w_min + 0.5 * (w0 - w_min) * (1.0 + math.cos(math.pi * q))


    def forward(
        self,
        logical_logits: Optional[torch.Tensor] = None,
        logical_labels: Optional[torch.Tensor] = None,
        logical_logits_x: Optional[torch.Tensor] = None,
        logical_logits_z: Optional[torch.Tensor] = None,
        label_x: Optional[torch.Tensor] = None,
        label_z: Optional[torch.Tensor] = None,
        pred_stabs: Optional[torch.Tensor] = None,
        true_stabs: Optional[torch.Tensor] = None,
        token_mask: Optional[torch.Tensor] = None,
        # ---- NEW: pass training progress so stabilizer weight can anneal ----
        step: Optional[int] = None,
        total_steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        支持兩種模式：
        1. 向後兼容模式：提供 logical_logits 和 logical_labels
        2. 分別的 X/Z 模式：提供 logical_logits_x, logical_logits_z, label_x, label_z
        
        logical_logits: (B,) or (B,1) - 向後兼容
        logical_labels: (B,) or (B,1) in {0,1} - 向後兼容
        
        logical_logits_x: (B,) or (B,1) - X 型邏輯錯誤 logits
        logical_logits_z: (B,) or (B,1) - Z 型邏輯錯誤 logits
        label_x: (B,) or (B,1) in {0,1} - X 型邏輯錯誤標籤
        label_z: (B,) or (B,1) in {0,1} - Z 型邏輯錯誤標籤

        pred_stabs / true_stabs (optional):
          - any shape, but must match each other.

        token_mask (optional):
          - mask for stab prediction tokens
          - shape should be broadcastable to pred_stabs/true_stabs once flattened
        """
        # ---- logical head ----
        # 優先使用分別的 X/Z 模式
        if logical_logits_x is not None and logical_logits_z is not None:
            # 分別的 X/Z 模式
            logical_logits_x = logical_logits_x.view(-1)  # (B,)
            logical_logits_z = logical_logits_z.view(-1)  # (B,)
            
            if label_x is None or label_z is None:
                raise ValueError("If logical_logits_x and logical_logits_z are provided, label_x and label_z must also be provided")
            
            label_x = label_x.view(-1).float()  # (B,)
            label_z = label_z.view(-1).float()  # (B,)
            
            # 分別計算 X 和 Z 的損失
            l_loss_x = self._bce_logits(
                logits=logical_logits_x,
                targets=label_x,
                pos_weight=self.logical_pos_weight,
                mask=None,
            )
            
            l_loss_z = self._bce_logits(
                logits=logical_logits_z,
                targets=label_z,
                pos_weight=self.logical_pos_weight,
                mask=None,
            )
            
            # 總邏輯損失 = X 損失 + Z 損失
            l_loss = l_loss_x + l_loss_z
            
        elif logical_logits is not None and logical_labels is not None:
            # 向後兼容模式
            logical_logits = logical_logits.view(-1)  # (B,)
            logical_labels = logical_labels.view(-1).float()  # (B,)

            l_loss = self._bce_logits(
                logits=logical_logits,
                targets=logical_labels,
                pos_weight=self.logical_pos_weight,
                mask=None,
            )
            l_loss_x = l_loss / 2.0  # 假設 X 和 Z 各佔一半（用於統計）
            l_loss_z = l_loss / 2.0
        else:
            raise ValueError(
                "Must provide either (logical_logits, logical_labels) "
                "or (logical_logits_x, logical_logits_z, label_x, label_z)"
            )

        # ---- next-stabilizer head (optional) ----
        # Use a reference tensor to get device/dtype (could be from X/Z mode or legacy mode)
        ref_tensor = logical_logits_x if logical_logits_x is not None else logical_logits
        s_loss = ref_tensor.new_zeros(())  # scalar 0 on same device/dtype
        if pred_stabs is not None and true_stabs is not None:
            assert pred_stabs.shape == true_stabs.shape, "pred_stabs and true_stabs must have same shape"

            # flatten to (N_tokens,)
            p = pred_stabs.reshape(-1)
            t = true_stabs.reshape(-1).float()

            m = None
            if token_mask is not None:
                # broadcast/expand token_mask to pred_stabs shape, then flatten
                if token_mask.shape != pred_stabs.shape:
                    try:
                        token_mask_exp = token_mask.expand_as(pred_stabs)
                    except Exception:
                        raise ValueError(
                            f"token_mask shape {token_mask.shape} not broadcastable to pred_stabs {pred_stabs.shape}"
                        )
                else:
                    token_mask_exp = token_mask
                m = token_mask_exp.reshape(-1)

            s_loss = self._bce_logits(
                logits=p,
                targets=t,
                pos_weight=self.stab_pos_weight,
                mask=m,
            )

        # ---- NEW: scheduled stabilizer weight (anneal over training) ----
        w = self._scheduled_next_stab_weight(step=step, total_steps=total_steps)

        total = l_loss + w * s_loss

        stats = {
            "loss": total.detach(),
            "logical_loss": l_loss.detach(),
            "logical_loss_x": l_loss_x.detach() if isinstance(l_loss_x, torch.Tensor) else l_loss_x,
            "logical_loss_z": l_loss_z.detach() if isinstance(l_loss_z, torch.Tensor) else l_loss_z,
            "stab_loss": s_loss.detach() if isinstance(s_loss, torch.Tensor) else s_loss,
            "next_stab_weight": float(w),
            "next_stab_schedule": self.next_stab_schedule,
        }
        return total, stats
