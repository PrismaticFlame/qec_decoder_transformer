# ema.py ok done
from __future__ import annotations

from typing import Dict, Any, Optional
import torch
import torch.nn as nn


class EMA:
    """
    Exponential Moving Average (EMA) for model parameters.

    Paper-style update (AlphaQubit-like):
        shadow = (1 - alpha) * shadow + alpha * param

    Where:
        alpha = "parameter exponential moving average constant"
        (paper uses alpha = 0.0001)

    Usage:
        ema = EMA(model, alpha=1e-4)
        ...
        optimizer.step()
        ema.update(model)   # AFTER optimizer.step()
        ...
        ema.apply_to(model) # for eval
        ...
        ema.restore(model)
    """

    def __init__(
        self,
        model: nn.Module,
        alpha: float = 1e-4,          # <-- paper constant (new weight fraction)
        track_buffers: bool = False,  # usually False for transformer/QEC
    ):
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0,1), got {alpha}")
        self.alpha = float(alpha)
        self.track_buffers = bool(track_buffers) #用來決定模型是否要紀錄並更新數據的統計狀態 ex.不需要計算梯度，但需要被記錄下來的數值。最經典的例子就是 BatchNorm 中的 running_mean (移動平均值) 和 running_var (移動變異數)。

        self.shadow_params: Dict[str, torch.Tensor] = {}
        self.shadow_buffers: Dict[str, torch.Tensor] = {}

        self._backup_params: Optional[Dict[str, torch.Tensor]] = None #當你要測試 shadow 的效果時，必須先把「正在訓練中的權重」暫時搬出來放進 backup
        self._backup_buffers: Optional[Dict[str, torch.Tensor]] = None

        self._init_from_model(model)

    @torch.no_grad()  #只需copy, update但不用gradient，記得寫裝飾
    def _init_from_model(self, model: nn.Module):
        # Parameters
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow_params[name] = p.detach().clone()


    @torch.no_grad()
    def update(self, model: nn.Module):
        """
        Update EMA weights. Must be called AFTER optimizer.step().
        """
        a = self.alpha

        # Parameters
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue

            if name not in self.shadow_params:
                # in case new params appear (rare)
                self.shadow_params[name] = p.detach().clone()
                continue

            shadow = self.shadow_params[name]
            shadow.mul_(1.0 - a).add_(p.detach(), alpha=a)
            #x.add_(other, alpha=α)=="x = x + α * other"



    @torch.no_grad()
    def apply_to(self, model: nn.Module):
        """
        Overwrite model params/buffers with EMA shadow (for eval).
        Call restore() afterwards.
        """
        if self._backup_params is not None:
            raise RuntimeError("EMA.apply_to() called twice without restore().")

        self._backup_params = {}
        for name, p in model.named_parameters():
            if name in self.shadow_params:
                self._backup_params[name] = p.detach().clone()
                p.copy_(self.shadow_params[name])

        if self.track_buffers:
            self._backup_buffers = {}
            for name, b in model.named_buffers():
                if b is None:
                    continue
                if name in self.shadow_buffers:
                    self._backup_buffers[name] = b.detach().clone() #原本buffer先備份好
                    b.copy_(self.shadow_buffers[name]) #原本buffer先放shadow做validation

    @torch.no_grad()
    def restore(self, model: nn.Module): #訓練中短暫換權重去validation,之後要繼續train
        """
        Restore original model params/buffers after apply_to().
        """
        if self._backup_params is None:
            return

        for name, p in model.named_parameters():
            if name in self._backup_params:
                p.copy_(self._backup_params[name])
        self._backup_params = None

        if self.track_buffers and self._backup_buffers is not None:
            for name, b in model.named_buffers():
                if b is None:
                    continue
                if name in self._backup_buffers:
                    b.copy_(self._backup_buffers[name])
            self._backup_buffers = None

    def state_dict(self) -> Dict[str, Any]:
        """
        Save EMA state for checkpointing.
        """
        return {
            "alpha": self.alpha,
            "track_buffers": self.track_buffers,
            "shadow_params": self.shadow_params,
            "shadow_buffers": self.shadow_buffers,
        }

    def load_state_dict(self, state: Dict[str, Any]): #載入一個模型版本 發生在pretrain optimizer重新調參 
        """
        Load EMA state from checkpoint.
        """
        self.alpha = float(state.get("alpha", self.alpha))
        self.track_buffers = bool(state.get("track_buffers", self.track_buffers))

        sp = state.get("shadow_params", {})
        sb = state.get("shadow_buffers", {})

        self.shadow_params = {k: v.detach().clone() for k, v in sp.items()}
        self.shadow_buffers = {k: v.detach().clone() for k, v in sb.items()}

        # clear backups
        self._backup_params = None
        self._backup_buffers = None
