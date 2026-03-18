# dataset.py - Data loading for Google hardware data (trans7)
#
# Reads the raw .01 ASCII files from data/goog_dem_data/data_0301/:
#   events_from_b8.01  - detection events (hardware, finetuning/testing)
#   events.01          - detection events (DEM-simulated, pretraining)
#   meas_from_b8.01    - measurements (hardware, when available)
#   meas.01            - measurements (DEM-simulated, when available)
#   obs_flips_actual.01- actual logical error labels (hardware)
#   obs_flips.01       - simulated logical error labels (DEM)
#
# Layout JSON is built from the stim circuit (circuit_noisy.stim or circuit_ideal.stim)
# and cached alongside the data folder.
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

# -----------------------------------------------------------------------
# Low-level .01 file reader
# -----------------------------------------------------------------------


def read_01_file(path: str) -> np.ndarray:
    """
    Read an ASCII .01 file into an (N, D) int8 array.

    File format: each line is a shot, characters are '0' or '1'.
    Single-column files (e.g. obs_flips) produce shape (N, 1).
    """
    raw = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    if raw.size == 0:
        return np.zeros((0, 0), dtype=np.int8)
    # Strip carriage returns (Windows line endings)
    raw = raw[raw != ord("\r")]
    # Find column width from first newline
    newline_pos = np.where(raw == ord("\n"))[0]
    if newline_pos.size == 0:
        # Single line, no trailing newline
        return (raw - ord("0")).astype(np.int8).reshape(1, -1)
    D = int(newline_pos[0])
    if D == 0:
        return np.zeros((0, 0), dtype=np.int8)
    # Drop newline characters, reshape to (N, D)
    data = raw[raw != ord("\n")]
    N = data.size // D
    return (data[: N * D] - ord("0")).astype(np.int8).reshape(N, D)


# -----------------------------------------------------------------------
# Measurement reconstruction fallback
# -----------------------------------------------------------------------


def compute_meas_from_events(
    events: np.ndarray,
    stab_id: List[int],
    cycle_id: List[int],
) -> np.ndarray:
    """
    Reconstruct measurements from detection events via cumulative XOR per stabilizer.

    events: (N, D) int8
    Returns: (N, D) int8
    """
    N, D = events.shape
    num_stab = max(stab_id) + 1
    meas = np.zeros_like(events)
    for s in range(num_stab):
        det_indices = [i for i, sid in enumerate(stab_id) if sid == s]
        det_indices.sort(key=lambda i: cycle_id[i])
        if not det_indices:
            continue
        idx = np.array(det_indices)
        meas[:, idx] = np.cumsum(events[:, idx], axis=1) % 2
    return meas.astype(np.int8)


# -----------------------------------------------------------------------
# SyndromeDataset (works for pretraining and finetuning)
# -----------------------------------------------------------------------


class SyndromeDataset(Dataset):
    """
    PyTorch Dataset for QEC transformer training on Google hardware data.

    Parameters
    ----------
    events : (N, D) int8   - detection events
    labels : (N,) int8     - logical error labels (0 or 1)
    layout : dict          - layout JSON (num_stab, stab_id, cycle_id, x, y, ...)
    measurements : (N, D) int8 or None  - if None, reconstructed from events
    """

    def __init__(
        self,
        events: np.ndarray,
        labels: np.ndarray,
        layout: Dict[str, Any],
        measurements: Optional[np.ndarray] = None,
    ):
        super().__init__()

        events = np.asarray(events, dtype=np.int8)
        labels = np.asarray(labels).ravel().astype(np.int8)
        assert events.shape[0] == labels.shape[0], (
            f"events/labels shot count mismatch: {events.shape[0]} vs {labels.shape[0]}"
        )

        D = events.shape[1]
        assert D == layout["num_detectors"], (
            f"events columns {D} != layout num_detectors {layout['num_detectors']}"
        )

        self.num_shots = events.shape[0]
        self.num_detectors = D
        self.distance = int(layout["distance"])
        assert self.distance > 0

        # basis: 0=X, 1=Z — stored as a scalar int tensor included in every batch item
        # so the model can use it for basis-aware readout pooling direction.
        basis_str = str(layout.get("basis", "x")).upper()
        self.basis_idx = torch.tensor(0 if basis_str == "X" else 1, dtype=torch.long)

        stab_id_arr = np.asarray(layout["stab_id"], dtype=np.int64)
        cycle_id_arr = np.asarray(layout["cycle_id"], dtype=np.int64)

        self.stab_id = torch.from_numpy(stab_id_arr).long()  # (D,)
        self.cycle_id = torch.from_numpy(cycle_id_arr).long()  # (D,)
        self.num_stab = int(stab_id_arr.max()) + 1
        self.num_cycles = int(cycle_id_arr.max()) + 1

        # Measurements must match events shape (D = circuit_detectors).
        # load_folder strips trailing data-qubit columns from meas.01 to ensure alignment.
        if measurements is not None:
            meas = np.asarray(measurements, dtype=np.int8)
            if meas.shape != events.shape:
                # Mismatched shape — fall back to cumulative XOR reconstruction
                meas = compute_meas_from_events(
                    events, layout["stab_id"], layout["cycle_id"]
                )
        else:
            meas = compute_meas_from_events(
                events, layout["stab_id"], layout["cycle_id"]
            )

        self.events = torch.from_numpy(events).long()  # (N, D)
        self.meas = torch.from_numpy(meas).float()  # (N, D)
        self.labels = torch.from_numpy(labels).float()  # (N,)

        # stab_type: 1=on-basis, 0=off-basis
        if "stab_type" in layout:
            st = np.asarray(layout["stab_type"], dtype=np.int64)
        else:
            st = np.ones(self.num_stab, dtype=np.int64)
        self.stab_type = torch.from_numpy(st).long()  # (num_stab,)

        # Stabilizer xy coordinates for bias provider
        if "x" in layout and "y" in layout:
            x_coords = np.asarray(layout["x"], dtype=np.float32)
            y_coords = np.asarray(layout["y"], dtype=np.float32)
            self._x_coords = torch.from_numpy(x_coords)
            self._y_coords = torch.from_numpy(y_coords)
        else:
            self._x_coords = None
            self._y_coords = None

        # Build per-cycle index (T, S_max)
        self.T, self.S, self.cycle_index, self.cycle_pad_mask = self._build_cycle_index(
            self.stab_id, self.cycle_id
        )

        # Build stab_xy from a full cycle
        if self._x_coords is not None and self.S > 0:
            for t in range(self.T):
                if self.cycle_pad_mask[t].all():
                    idx_full = self.cycle_index[t]
                    break
            else:
                idx_full = self.cycle_index[0]
            x0 = self._x_coords[idx_full]
            y0 = self._y_coords[idx_full]
            self.stab_xy = torch.stack([x0, y0], dim=-1)  # (S, 2)
        else:
            self.stab_xy = None

    @staticmethod
    def _build_cycle_index(
        stab_id: torch.Tensor, cycle_id: torch.Tensor
    ) -> Tuple[int, int, torch.Tensor, torch.Tensor]:
        T = int(cycle_id.max().item()) + 1
        indices_per_t = []
        S_max = 0
        for t in range(T):
            idx_t = torch.nonzero(cycle_id == t, as_tuple=False).view(-1)
            if idx_t.numel() > 0:
                idx_t = idx_t[torch.argsort(stab_id[idx_t])]
            indices_per_t.append(idx_t)
            S_max = max(S_max, idx_t.numel())
        S = int(S_max)
        if S == 0:
            z = torch.zeros((T, 0), dtype=torch.long)
            return T, S, z, z.bool()
        padded, masks = [], []
        for idx_t in indices_per_t:
            n = idx_t.numel()
            if n < S:
                pad = torch.zeros(S - n, dtype=torch.long)
                padded.append(torch.cat([idx_t, pad]))
                masks.append(
                    torch.cat(
                        [
                            torch.ones(n, dtype=torch.bool),
                            torch.zeros(S - n, dtype=torch.bool),
                        ]
                    )
                )
            else:
                padded.append(idx_t[:S])
                masks.append(torch.ones(S, dtype=torch.bool))
        return T, S, torch.stack(padded).long(), torch.stack(masks)

    def __len__(self) -> int:
        return self.num_shots

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        syndrome = self.events[idx].long()  # (D,)
        measurements = self.meas[idx]  # (D,)
        label = self.labels[idx]  # scalar

        # Next-stab targets (T-1, S)
        ci = self.cycle_index  # (T, S)
        synd_ts = syndrome[ci]  # (T, S)
        mask_ts = self.cycle_pad_mask  # (T, S)
        true_stabs = synd_ts[1:].float()  # (T-1, S)
        token_mask = mask_ts[1:].clone()  # (T-1, S)

        batch = {
            "syndrome": syndrome,
            "measurements": measurements,
            "stab_id": self.stab_id,
            "cycle_id": self.cycle_id,
            "stab_type": self.stab_type,
            "logical_labels": label.unsqueeze(0),
            "label": label.unsqueeze(0),
            "true_stabs": true_stabs,
            "token_mask": token_mask,
            "cycle_index": self.cycle_index,
            "cycle_pad_mask": self.cycle_pad_mask,
            "basis_idx": self.basis_idx,  # scalar: 0=X, 1=Z
        }
        if self.stab_xy is not None:
            batch["stab_xy"] = self.stab_xy
        return batch


# -----------------------------------------------------------------------
# Data-source resolution helpers
# -----------------------------------------------------------------------


def _resolve_file(folder: Path, *candidates: str) -> Optional[Path]:
    """Return the first candidate filename that exists in folder, or None."""
    for name in candidates:
        p = folder / name
        if p.exists():
            return p
    return None


def load_folder(
    folder: Path,
    layout: Dict[str, Any],
    *,
    prefer_hardware: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Load (events, labels, measurements) from a data folder.

    prefer_hardware=True  -> prefer events_from_b8.01 / obs_flips_actual.01
    prefer_hardware=False -> prefer events.01 / obs_flips.01  (pretraining / DEM)

    Measurements fall back to None (dataset will reconstruct via cumXOR).
    """
    if prefer_hardware:
        ev_cands = ("events_from_b8.01", "events.01")
        lb_cands = ("obs_flips_actual.01", "obs_flips.01")
        ms_cands = ("meas_from_b8.01", "meas.01")
    else:
        ev_cands = ("events.01", "events_from_b8.01")
        lb_cands = ("obs_flips.01", "obs_flips_actual.01")
        ms_cands = ("meas.01", "meas_from_b8.01")

    ev_path = _resolve_file(folder, *ev_cands)
    lb_path = _resolve_file(folder, *lb_cands)
    if ev_path is None or lb_path is None:
        raise FileNotFoundError(
            f"Cannot find events or labels in {folder}.\n"
            f"  Looked for events: {ev_cands}\n"
            f"  Looked for labels: {lb_cands}"
        )

    # Inject basis into layout from folder name (bX or bZ) so SyndromeDataset can include
    # it in each batch for basis-aware readout. Folder names always contain _bX_ or _bZ_.
    folder_name = Path(folder).name
    if "_bX_" in folder_name or folder_name.startswith("surface_code_bX"):
        layout = {**layout, "basis": "X"}
    elif "_bZ_" in folder_name or folder_name.startswith("surface_code_bZ"):
        layout = {**layout, "basis": "Z"}

    events = read_01_file(str(ev_path))  # (N, D)
    labels = read_01_file(str(lb_path)).ravel()  # (N,)
    assert events.shape[0] == labels.shape[0], (
        f"Shot mismatch: events={events.shape[0]}, labels={labels.shape[0]} in {folder}"
    )

    ms_path = _resolve_file(folder, *ms_cands)
    measurements = None
    if ms_path is not None:
        measurements = read_01_file(str(ms_path))
        D_events = events.shape[1]
        if measurements.shape[1] != D_events:
            # meas.01 = (stabilizer measurements across all rounds) + (final data qubit readouts)
            # The data qubit readouts are always appended as the last data_qubits columns.
            # Strip them so the remaining stabilizer columns align 1:1 with events.
            n_data = measurements.shape[1] - D_events
            if n_data > 0:
                measurements = measurements[:, :-n_data]
            if measurements.shape[1] != D_events:
                measurements = (
                    None  # unexpected shape, fall back to cumXOR in SyndromeDataset
                )

    return events, labels, measurements


def make_train_val_split(
    events: np.ndarray,
    labels: np.ndarray,
    measurements: Optional[np.ndarray],
    *,
    n_train: int,
    n_val: int,
    seed: int = 42,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]],
    Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]],
]:
    """
    Split (events, labels, measurements) into train and val sets.
    Uses the first n_train shots for training and the next n_val for validation.
    """
    N = events.shape[0]
    assert n_train + n_val <= N, f"n_train={n_train} + n_val={n_val} > N={N}"
    rng = np.random.RandomState(seed)
    idx = rng.permutation(N)
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]

    def _split(arr):
        if arr is None:
            return None, None
        return arr[train_idx], arr[val_idx]

    ev_tr, ev_val = _split(events)
    lb_tr, lb_val = _split(labels)
    ms_tr, ms_val = _split(measurements)

    return (ev_tr, lb_tr, ms_tr), (ev_val, lb_val, ms_val)


# -----------------------------------------------------------------------
# Multi-folder dataset (aggregate several round folders)
# -----------------------------------------------------------------------


class MultiRoundDataset(Dataset):
    """
    Aggregates shots from multiple round folders for the same distance.

    Different round counts have different num_detectors (D = num_stab * rounds).
    Use make_loader() which installs a GroupedBatchSampler to ensure every batch
    contains only samples with the same D (same round count folder), so the model
    never receives a mixed-shape batch.
    """

    def __init__(self, datasets: List[SyndromeDataset]):
        assert datasets, "datasets list must not be empty"
        self._datasets = datasets
        # Build cumulative index for __getitem__
        self._cumlen = np.cumsum([len(d) for d in datasets])

    def __len__(self) -> int:
        return int(self._cumlen[-1])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ds_idx = int(np.searchsorted(self._cumlen, idx, side="right"))
        offset = 0 if ds_idx == 0 else int(self._cumlen[ds_idx - 1])
        return self._datasets[ds_idx][idx - offset]

    def make_grouped_sampler(
        self,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 0,
    ) -> "GroupedBatchSampler":
        """Return a sampler that keeps same-D samples together in each batch.

        Scales batch size inversely with num_cycles so activation memory stays
        roughly constant across round groups (more cycles → smaller batch).
        The reference is the minimum num_cycles group (T_min), which gets
        batch_size; longer groups are scaled by T_min / T, clamped to ≥ 1.
        """
        groups = []
        t_counts = []
        for ds_idx, ds in enumerate(self._datasets):
            start = 0 if ds_idx == 0 else int(self._cumlen[ds_idx - 1])
            groups.append(list(range(start, start + len(ds))))
            t_counts.append(ds.num_cycles)
        t_min = max(1, min(t_counts))
        batch_sizes = [max(1, int(round(batch_size * t_min / t))) for t in t_counts]
        return GroupedBatchSampler(
            groups,
            batch_sizes=batch_sizes,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=seed,
        )

    def make_distributed_grouped_sampler(
        self,
        batch_size: int,
        rank: int,
        world_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 0,
    ) -> "DistributedGroupedBatchSampler":
        """Return a distributed sampler that partitions same-D groups across ranks."""
        groups = []
        t_counts = []
        for ds_idx, ds in enumerate(self._datasets):
            start = 0 if ds_idx == 0 else int(self._cumlen[ds_idx - 1])
            groups.append(list(range(start, start + len(ds))))
            t_counts.append(ds.num_cycles)
        t_min = max(1, min(t_counts))
        batch_sizes = [max(1, int(round(batch_size * t_min / t))) for t in t_counts]
        return DistributedGroupedBatchSampler(
            groups,
            batch_sizes=batch_sizes,
            rank=rank,
            world_size=world_size,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=seed,
        )


class GroupedBatchSampler(Sampler):
    """
    Yields batches where all indices come from the same group (same round count,
    same D). Groups are visited in random order each epoch when shuffle=True.

    batch_sizes: per-group batch sizes (scaled down for long-round groups to
    keep activation memory roughly constant across different T values).
    """

    def __init__(
        self,
        groups: List[List[int]],
        batch_sizes: List[int],
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 0,
    ):
        assert len(groups) == len(batch_sizes)
        self._groups = groups
        self._batch_sizes = batch_sizes
        self._shuffle = shuffle
        self._drop_last = drop_last
        self._seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __iter__(self):
        rng = np.random.RandomState(self._seed + self._epoch)
        all_batches = []
        for group, bs in zip(self._groups, self._batch_sizes):
            indices = list(group)
            if self._shuffle:
                rng.shuffle(indices)
            for start in range(0, len(indices), bs):
                batch = indices[start : start + bs]
                if self._drop_last and len(batch) < bs:
                    continue
                all_batches.append(batch)
        if self._shuffle:
            rng.shuffle(all_batches)
        yield from all_batches

    def __len__(self) -> int:
        total = 0
        for group, bs in zip(self._groups, self._batch_sizes):
            n = len(group)
            if self._drop_last:
                total += n // bs
            else:
                total += (n + bs - 1) // bs
        return total


class DistributedGroupedBatchSampler(Sampler):
    """
    Distributed version of GroupedBatchSampler for multi-GPU training.

    Partitions each group's indices across world_size ranks so every rank
    gets a disjoint subset. Each rank then forms batches from its own subset
    while keeping same-D grouping intact.
    """

    def __init__(
        self,
        groups: List[List[int]],
        batch_sizes: List[int],
        rank: int,
        world_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 0,
    ):
        assert len(groups) == len(batch_sizes)
        self._groups = groups
        self._batch_sizes = batch_sizes
        self._rank = rank
        self._world_size = world_size
        self._shuffle = shuffle
        self._drop_last = drop_last
        self._seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __iter__(self):
        rng = np.random.RandomState(self._seed + self._epoch)
        all_batches = []
        for group, bs in zip(self._groups, self._batch_sizes):
            indices = list(group)
            # All ranks shuffle with the same seed so partitioning is consistent
            if self._shuffle:
                rng.shuffle(indices)
            # Partition: this rank gets indices[rank::world_size]
            rank_indices = indices[self._rank :: self._world_size]
            for start in range(0, len(rank_indices), bs):
                batch = rank_indices[start : start + bs]
                if self._drop_last and len(batch) < bs:
                    continue
                all_batches.append(batch)
        # Shuffle batch order (with rank-specific seed so ranks differ)
        if self._shuffle:
            batch_rng = np.random.RandomState(self._seed + self._epoch + self._rank)
            batch_rng.shuffle(all_batches)
        yield from all_batches

    def __len__(self) -> int:
        total = 0
        for group, bs in zip(self._groups, self._batch_sizes):
            n = len(group) // self._world_size
            if self._drop_last:
                total += n // bs
            else:
                total += (n + bs - 1) // bs
        return total


# -----------------------------------------------------------------------
# DataLoader factory
# -----------------------------------------------------------------------


def make_loader(
    dataset: Dataset,
    batch_size: int,
    *,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = True,
    seed: int = 0,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    distributed = world_size > 1

    if isinstance(dataset, MultiRoundDataset):
        if distributed:
            sampler = dataset.make_distributed_grouped_sampler(
                batch_size=batch_size,
                rank=rank,
                world_size=world_size,
                shuffle=shuffle,
                drop_last=drop_last,
                seed=seed,
            )
        else:
            sampler = dataset.make_grouped_sampler(
                batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, seed=seed
            )
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    if distributed:
        from torch.utils.data.distributed import DistributedSampler

        dist_sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=seed,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=dist_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


# -----------------------------------------------------------------------
# Seed utility
# -----------------------------------------------------------------------


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
