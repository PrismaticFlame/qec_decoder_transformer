# src/dataset.py
from __future__ import annotations

import json
from typing import Dict, Optional, Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SyndromeDataset(Dataset):
    """
    Dataset for QEC transformer (AlphaQubit style hard/soft input).

    samples: (N, L)
      - hard: 0/1 detector events
      - soft: continuous values (LLR/prob/analog surrogate)
    labels: (N,) or (N,1)

    layout.json required keys:
      - num_detectors: int (L)
      - stab_id: list[int] length L
      - cycle_id: list[int] length L
      - distance: int
    """

    def __init__(
        self,
        samples: np.ndarray,
        labels: np.ndarray,
        layout_json_path: str,
        *,
        input_mode: str = "hard",              # "hard" or "soft"
        mask_last_cycle: bool = False,         # 若要把最後一輪 detector 全遮掉（避免 label leak/terminal）
        custom_token_mask: Optional[np.ndarray] = None,  # (L,) bool, 1=keep
        leakage_mask: Optional[np.ndarray] = None,  # (N, L) bool, True=kept, False=leaked
        measurements: Optional[np.ndarray] = None,  # (N, L) int8 - pre-computed measurements
    ):
        super().__init__()

        samples = np.asarray(samples)
        labels = np.asarray(labels)

        if samples.ndim == 1:
            samples = samples.reshape(1, -1)
        
        # 支持兩種格式：
        # 1. labels 是 (N, 1) 或 (N,) - 單個邏輯標籤（向後兼容）
        # 2. labels 是 (N, 2) - X 和 Z 邏輯標籤
        if labels.ndim == 1:
            labels = labels.reshape(-1, 1)
        elif labels.ndim == 2 and labels.shape[1] == 1:
            # 保持 (N, 1) 格式
            pass
        elif labels.ndim == 2 and labels.shape[1] >= 2:
            # (N, K) 格式，K >= 2，取前兩個作為 X 和 Z
            labels = labels[:, :2]  # (N, 2)
        else:
            raise ValueError(f"labels shape {labels.shape} not supported. Expected (N,), (N,1), or (N,2)")

        assert samples.shape[0] == labels.shape[0], "shots mismatch"

        with open(layout_json_path, "r") as f:
            layout = json.load(f)

        L_expected = int(layout["num_detectors"])
        assert samples.shape[1] == L_expected, (
            f"sample L={samples.shape[1]} != layout num_detectors={L_expected}"
        )

        stab_id = np.asarray(layout["stab_id"], dtype=np.int64)
        cycle_id = np.asarray(layout["cycle_id"], dtype=np.int64)
        assert stab_id.shape[0] == L_expected and cycle_id.shape[0] == L_expected

        # shared ids
        self.stab_id = torch.from_numpy(stab_id).long()      # (L,)
        self.cycle_id = torch.from_numpy(cycle_id).long()    # (L,)
        
        # Extract coordinates for bias provider (if available in layout)
        # We need per-cycle stabilizer coordinates (S, 2)
        if "x" in layout and "y" in layout:
            x_coords = np.asarray(layout["x"], dtype=np.float32)
            y_coords = np.asarray(layout["y"], dtype=np.float32)
            # Build stab_xy per cycle: for each cycle, extract coordinates of stabilizers
            # After building cycle_index, we'll extract coordinates for the first cycle
            # (all cycles should have the same stabilizer positions)
            self._x_coords = torch.from_numpy(x_coords)  # (L,)
            self._y_coords = torch.from_numpy(y_coords)  # (L,)
        else:
            self._x_coords = None
            self._y_coords = None

        # meta (model init 用)
        self.num_stab = int(stab_id.max()) + 1
        self.num_cycles = int(cycle_id.max()) + 1
        self.distance = int(layout.get("distance", 0))
        assert self.distance > 0, "layout.json must contain positive 'distance'"

        # ------------------------------------------------
        # detector-level mask (L,) : 用於遮掉 syndrome input
        # ------------------------------------------------
        det_mask = np.ones((L_expected,), dtype=bool)

        if mask_last_cycle:
            max_c = int(cycle_id.max())
            det_mask &= (cycle_id != max_c)

        if custom_token_mask is not None:
            custom_token_mask = np.asarray(custom_token_mask, dtype=bool)
            assert custom_token_mask.shape == (L_expected,), "custom_token_mask must be shape (L,)"
            det_mask &= custom_token_mask

        # Store leakage mask if provided (per-sample masking)
        self.leakage_mask = None
        if leakage_mask is not None:
            leakage_mask = np.asarray(leakage_mask, dtype=bool)
            if leakage_mask.shape == (samples.shape[0], L_expected):
                # Per-sample leakage mask
                self.leakage_mask = torch.from_numpy(leakage_mask)  # (N, L) bool
            elif leakage_mask.shape == (L_expected,):
                # Global leakage mask (same for all samples)
                self.leakage_mask = torch.from_numpy(leakage_mask).unsqueeze(0).expand(samples.shape[0], -1)  # (N, L)
            else:
                raise ValueError(f"leakage_mask must be shape (N, L) or (L,), got {leakage_mask.shape}")

        self.det_token_mask = torch.from_numpy(det_mask)  # (L,) bool (True=keep)

        # store both representations
        self.dets_hard = torch.from_numpy(np.rint(samples).astype(np.int64))    # (N,L)
        self.dets_soft = torch.from_numpy(samples.astype(np.float32))           # (N,L)

        # Store measurements (pre-computed from data generation, or compute fallback)
        if measurements is not None:
            measurements = np.asarray(measurements)
            assert measurements.shape == samples.shape, (
                f"measurements shape {measurements.shape} != samples shape {samples.shape}"
            )
            self.meas_hard = torch.from_numpy(np.rint(measurements).astype(np.int64))  # (N, L)
        else:
            # Fallback: compute from detection events via cumulative XOR per stabilizer
            self.meas_hard = self._compute_meas_fallback(self.dets_hard, stab_id, cycle_id)

        # Load stab_type from layout (1=on-basis, 0=off-basis)
        if "stab_type" in layout:
            stab_type_per_stab = np.asarray(layout["stab_type"], dtype=np.int64)  # (num_stab,)
            self.stab_type = torch.from_numpy(stab_type_per_stab).long()
        else:
            # Fallback: all stabilizers treated as on-basis
            self.stab_type = torch.ones(self.num_stab, dtype=torch.long)

        # 處理邏輯標籤：如果只有一個標籤，則 X 和 Z 使用相同的標籤（向後兼容）
        if labels.shape[1] == 1:
            # 單個標籤：X 和 Z 使用相同的標籤
            self.labels_x = torch.from_numpy(labels[:, 0]).float()  # (N,)
            self.labels_z = torch.from_numpy(labels[:, 0]).float()  # (N,)
        else:
            # 兩個標籤：第一個是 X，第二個是 Z
            self.labels_x = torch.from_numpy(labels[:, 0]).float()  # (N,)
            self.labels_z = torch.from_numpy(labels[:, 1]).float()  # (N,)
        
        # 向後兼容：保留單個 label
        self.labels = self.labels_x  # (N,)

        input_mode = input_mode.lower().strip()
        self.input_mode = input_mode
        if input_mode == "hard":
            self.num_shots, self.num_detectors = self.dets_hard.shape
        else:
            self.num_shots, self.num_detectors = self.dets_soft.shape
        

        # ------------------------------------------------
        # build (T,S) cycle_index ONCE for next-stab labels
        # ------------------------------------------------
        self.T, self.S, self.cycle_index, self.cycle_pad_mask = self._build_cycle_index(self.stab_id, self.cycle_id)
        # cycle_index: (T, S) long — padded to S_max
        # cycle_pad_mask: (T, S) bool — True = real token, False = padding

        # Build stab_xy for bias provider: use the cycle with most stabilizers
        # (S_max), extracting coordinates only for real (non-padded) positions.
        if self._x_coords is not None and self._y_coords is not None and self.S > 0:
            # Find first cycle that has S stabilizers (a full cycle)
            for t in range(self.T):
                if self.cycle_pad_mask[t].all():
                    idx_full = self.cycle_index[t]  # (S,)
                    break
            else:
                # Fallback: use first cycle, only real positions
                idx_full = self.cycle_index[0]

            x_0 = self._x_coords[idx_full]  # (S,)
            y_0 = self._y_coords[idx_full]  # (S,)
            self.stab_xy = torch.stack([x_0, y_0], dim=-1)  # (S, 2)
        else:
            self.stab_xy = None  # Will fall back to ManhattanDistanceBias if not available

    @staticmethod
    def _compute_meas_fallback(
        dets_hard: torch.Tensor,
        stab_id: np.ndarray,
        cycle_id: np.ndarray,
    ) -> torch.Tensor:
        """Compute measurements from detection events via cumulative XOR per stabilizer."""
        N, L = dets_hard.shape
        meas = torch.zeros_like(dets_hard)
        num_stab = int(stab_id.max()) + 1
        for s in range(num_stab):
            det_indices = [i for i in range(L) if stab_id[i] == s]
            det_indices.sort(key=lambda i: cycle_id[i])
            if not det_indices:
                continue
            idx = torch.tensor(det_indices, dtype=torch.long)
            meas[:, idx] = torch.cumsum(dets_hard[:, idx], dim=1) % 2
        return meas

    @staticmethod
    def _build_cycle_index(stab_id: torch.Tensor, cycle_id: torch.Tensor):
        """
        Build indices mapping from flat L tokens -> per-cycle (T,S) tokens.
        For each cycle t, select positions where cycle_id == t, then sort by stab_id
        so the per-cycle ordering is consistent across time.
        Cycles with fewer tokens are padded to S_max using index 0 (masked out
        downstream via pad_mask). This preserves all detector information.

        Returns:
            T: number of cycles
            S: max stabilizers per cycle (S_max)
            indices: (T, S) long tensor of flat indices into syndrome
            pad_mask: (T, S) bool tensor, True = real token, False = padding
        """
        if stab_id.dim() != 1 or cycle_id.dim() != 1:
            raise ValueError(f"stab_id and cycle_id must be 1D, got {stab_id.shape}, {cycle_id.shape}")
        if stab_id.numel() != cycle_id.numel():
            raise ValueError(f"stab_id and cycle_id must have same length, got {stab_id.numel()} vs {cycle_id.numel()}")

        L = cycle_id.numel()
        # cycles assumed 0..T-1
        T = int(cycle_id.max().item()) + 1
        if T <= 0:
            raise ValueError(f"Invalid T computed from cycle_id.max(): T={T}")

        indices_per_t = []
        S_max = 0

        for t in range(T):
            idx_t = torch.nonzero(cycle_id == t, as_tuple=False).view(-1)
            if idx_t.numel() > 0:
                # sort within cycle by stab_id to make stable order
                idx_t = idx_t[torch.argsort(stab_id[idx_t])]
            indices_per_t.append(idx_t)
            S_max = max(S_max, idx_t.numel())

        S = int(S_max)
        if S <= 0:
            indices = torch.zeros((T, 0), dtype=torch.long, device=cycle_id.device)
            pad_mask = torch.zeros((T, 0), dtype=torch.bool, device=cycle_id.device)
            return T, S, indices, pad_mask

        # Pad each cycle to S_max; padded positions use index 0 (arbitrary,
        # masked out later) and are marked False in pad_mask.
        padded = []
        masks = []
        for idx_t in indices_per_t:
            n = idx_t.numel()
            if n < S:
                pad = torch.zeros(S - n, dtype=torch.long, device=cycle_id.device)
                padded.append(torch.cat([idx_t, pad], dim=0))
                mask_t = torch.cat([
                    torch.ones(n, dtype=torch.bool, device=cycle_id.device),
                    torch.zeros(S - n, dtype=torch.bool, device=cycle_id.device),
                ])
                masks.append(mask_t)
            else:
                padded.append(idx_t[:S])
                masks.append(torch.ones(S, dtype=torch.bool, device=cycle_id.device))

        indices = torch.stack(padded, dim=0).long()  # (T, S)
        pad_mask = torch.stack(masks, dim=0)          # (T, S)

        return T, S, indices, pad_mask

    def __len__(self):
        return self.num_shots

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # (L,)
        if self.input_mode == "hard":
            syndrome = self.dets_hard[idx].long()
            syndrome = syndrome * self.det_token_mask.long()
        else:
            syndrome = self.dets_soft[idx].float()
            syndrome = syndrome * self.det_token_mask.float()
        
        # Apply leakage mask if available
        if self.leakage_mask is not None:
            leak_mask = self.leakage_mask[idx]  # (L,) bool
            if self.input_mode == "hard":
                syndrome = syndrome * leak_mask.long()
            else:
                syndrome = syndrome * leak_mask.float()

        logical_labels = self.labels[idx].view(-1)  # (1,)
        label_x = self.labels_x[idx]  # scalar
        label_z = self.labels_z[idx]  # scalar

        # ----------------------------
        # next-stab labels (T-1,S)
        # ----------------------------
        ci = self.cycle_index  # (T,S)
        synd_ts = syndrome[ci]  # (T,S)
        mask_ts = self.det_token_mask[ci]  # (T,S) bool

        # Incorporate cycle padding mask: padded positions should not contribute
        # to the next-stab loss
        cycle_pm = self.cycle_pad_mask  # (T, S) bool
        mask_ts = mask_ts & cycle_pm

        # predict next cycle: target = cycles 1..T-1
        true_stabs = synd_ts[1:, :]                       # (T-1,S)
        token_mask = mask_ts[1:, :].clone()               # (T-1,S) bool

        # Apply leakage mask to token_mask if available
        if self.leakage_mask is not None:
            leak_mask_ts = self.leakage_mask[idx][ci]  # (T, S) bool
            leak_mask_next = leak_mask_ts[1:, :]  # (T-1, S) bool
            token_mask = token_mask & leak_mask_next  # Combine with existing mask

        # ensure true_stabs is float for BCE-with-logits
        true_stabs = true_stabs.float() if true_stabs.dtype != torch.float32 else true_stabs

        # Measurements (flat, same layout as syndrome)
        measurements = self.meas_hard[idx].float()  # (L,)

        batch = {
            # model inputs
            "syndrome": syndrome,              # (L,) detection events
            "measurements": measurements,      # (L,) pre-computed measurements
            "stab_id": self.stab_id,           # (L,)
            "cycle_id": self.cycle_id,         # (L,)
            "stab_type": self.stab_type,       # (num_stab,) 1=on-basis, 0=off-basis

            # labels (new + legacy)
            "logical_labels": logical_labels,  # (1,)
            "label": logical_labels,           # legacy key (keep old code alive)
            "label_x": label_x,                # (,) X 型邏輯錯誤標籤
            "label_z": label_z,                # (,) Z 型邏輯錯誤標籤

            # next-stab task
            "true_stabs": true_stabs,          # (T-1,S) float
            "token_mask": token_mask,          # (T-1,S) bool

            # keep detector-level mask too (debug/optional)
            "det_token_mask": self.det_token_mask,  # (L,) bool

            # For AttentionBiasProvider
            "cycle_index": self.cycle_index,   # (T, S) indices for per-cycle mapping
            "cycle_pad_mask": self.cycle_pad_mask,  # (T, S) bool — True = real, False = pad
        }
        
        # Add stab_xy if available (for AttentionBiasProvider)
        if self.stab_xy is not None:
            # Map from unique stab_ids to per-cycle stabilizers
            # For each cycle, we need (S, 2) coordinates
            # Since cycle_index maps (T, S) -> L indices, we can extract coordinates
            # But we need to map from L indices to unique stab positions
            # For now, provide the full stab_xy and let the model handle per-cycle extraction
            batch["stab_xy"] = self.stab_xy  # (num_stab, 2)
        
        return batch


def make_loader(
    dataset: Dataset,
    batch_size: Optional[int] = None,
    cfg: Optional[Any] = None,
    *,
    shuffle: bool,
    num_workers: Optional[int] = None,
    pin_memory: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """
    Compatible with BOTH call styles:
      1) make_loader(ds, batch_size=64, shuffle=True)
      2) make_loader(ds, cur_bs, cfg, shuffle=True)   # cfg is optional; if provided, we may read worker hints
    """
    if batch_size is None:
        raise ValueError("batch_size is required")

    if num_workers is None:
        # if cfg provides num_workers, use it; else default 0
        num_workers = int(getattr(cfg, "num_workers", 0)) if cfg is not None else 0

    pin_memory = bool(getattr(cfg, "pin_memory", pin_memory)) if cfg is not None else bool(pin_memory)
    drop_last = bool(getattr(cfg, "drop_last", drop_last)) if cfg is not None else bool(drop_last)

    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def set_seed(seed: int):  #np.random.default_rng(seed)不用這個嗎
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # optional: determinism (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
