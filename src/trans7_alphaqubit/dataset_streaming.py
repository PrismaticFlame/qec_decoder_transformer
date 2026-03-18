"""
dataset_streaming.py - Rolling window HDF5 dataset for pretraining (trans7)

ChunkedHDF5Dataset reads a monolithic HDF5 file built by data/data_random_sample.py
and yields one MultiRoundDataset chunk at a time. Each chunk contains chunk_size
samples drawn from the pre-split sample_index_train or sample_index_val stored in
the HDF5 file.

Within each chunk, samples are grouped by (basis, distance, rounds) so that
GroupedBatchSampler can produce D-homogeneous batches, exactly as the existing
folder-based loading path does.

Usage:
    from dataset_streaming import ChunkedHDF5Dataset, get_reference_layout
    from dataset import make_loader

    train_ds = ChunkedHDF5Dataset("data/trans7_data/pretrain.h5", split="train", distance=3)
    val_ds   = ChunkedHDF5Dataset("data/trans7_data/pretrain.h5", split="val",   distance=3)
    for chunk in train_ds:
        loader = make_loader(chunk, batch_size=256, shuffle=True)
        for batch in loader:
            ...
"""

from __future__ import annotations

import json
import math
import queue
import threading
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np
from dataset import MultiRoundDataset, SyndromeDataset

# -----------------------------------------------------------------------
# Layout helpers
# -----------------------------------------------------------------------


def _reconstruct_layout(grp) -> dict:
    """Reconstruct a layout dict from an HDF5 group's attributes."""
    layout = {
        "distance": int(grp.attrs["distance"]),
        "basis": str(grp.attrs["basis"]),
        "num_stab": int(grp.attrs["num_stab"]),
        "num_cycles": int(grp.attrs["num_cycles"]),
        "num_detectors": int(grp.attrs["num_detectors"]),
        "tokens_per_cycle": int(grp.attrs["tokens_per_cycle"]),
    }
    for key in ("stab_id", "cycle_id", "x", "y", "stab_type"):
        if key in grp.attrs:
            layout[key] = json.loads(grp.attrs[key])
    return layout


def get_reference_layout(h5_path: str | Path, distance: int) -> dict:
    """
    Return the layout dict for the max-round group matching `distance`.

    This is used to build the model (num_stab, num_cycles, grid coordinates)
    before any training data is loaded.
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is required. Install with: pip install h5py")

    with h5py.File(Path(h5_path), "r") as hf:
        raw_keys = hf["group_keys"][:]
        group_keys = [
            k.decode("utf-8") if isinstance(k, bytes) else str(k) for k in raw_keys
        ]
        max_rounds = -1
        ref_key = None
        for key in group_keys:
            grp = hf[key]
            if int(grp.attrs["distance"]) == distance:
                rounds = int(grp.attrs["rounds"])
                if rounds > max_rounds:
                    max_rounds = rounds
                    ref_key = key

        if ref_key is None:
            raise ValueError(f"No groups with distance={distance} found in {h5_path}")
        return _reconstruct_layout(hf[ref_key])


# -----------------------------------------------------------------------
# ChunkedHDF5Dataset
# -----------------------------------------------------------------------


class ChunkedHDF5Dataset:
    """
    Iterable over MultiRoundDataset chunks loaded from a monolithic HDF5 file.

    Parameters
    ----------
    h5_path : str or Path
        Path to the HDF5 file produced by data/data_random_sample.py.
    split : str
        Which split to use: ``"train"`` or ``"val"``.
    chunk_size : int
        Number of samples to load into RAM per chunk (rolling window size).
    distance : int or None
        If given, only samples from groups with this distance are included.
        Use this to train a per-distance model from a multi-distance HDF5 file.
    seed : int
        Seed for optionally re-shuffling the sample_index at construction time.
    shuffle : bool
        Whether to re-shuffle the sample_index with `seed` before iterating.
        Set to False to reproduce a fixed order (e.g. for reproducible validation).
    """

    def __init__(
        self,
        h5_path: str | Path,
        split: str = "train",
        chunk_size: int = 50_000,
        distance: Optional[int] = None,
        seed: int = 42,
        shuffle: bool = True,
    ):
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py is required. Install with: pip install h5py")

        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")

        self.h5_path = Path(h5_path)
        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")

        self.chunk_size = chunk_size
        self.distance = distance
        self.split = split

        index_key = f"sample_index_{split}"

        # Load the lightweight index arrays into RAM — tiny regardless of dataset size.
        with h5py.File(self.h5_path, "r") as hf:
            sample_index = hf[index_key][:]  # (M, 2) int32
            raw_keys = hf["group_keys"][:]
            self._group_keys = [
                k.decode("utf-8") if isinstance(k, bytes) else str(k) for k in raw_keys
            ]

            # Build per-group distance lookup so we can filter without re-opening.
            self._group_distances = {}
            for g_idx, key in enumerate(self._group_keys):
                self._group_distances[g_idx] = int(hf[key].attrs["distance"])

        # Filter sample_index to the requested distance if specified.
        if distance is not None:
            valid_groups = {
                g_idx for g_idx, d in self._group_distances.items() if d == distance
            }
            if not valid_groups:
                raise ValueError(
                    f"No groups with distance={distance} found in {self.h5_path}"
                )
            mask = np.isin(sample_index[:, 0], list(valid_groups))
            sample_index = sample_index[mask]

        self._sample_index = sample_index

        if shuffle:
            rng = np.random.RandomState(seed)
            rng.shuffle(self._sample_index)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_samples(self) -> int:
        return len(self._sample_index)

    @property
    def num_chunks(self) -> int:
        return math.ceil(self.total_samples / self.chunk_size)

    # ------------------------------------------------------------------
    # Internal: read one group's rows from HDF5
    # ------------------------------------------------------------------

    def _read_group(self, hf, g_idx: int, local_indices: list):
        """Read events/labels/measurements for a group, sorted for HDF5 efficiency."""
        key = self._group_keys[g_idx]
        grp = hf[key]
        layout = _reconstruct_layout(grp)

        order = np.argsort(local_indices)
        sorted_local = np.array(local_indices, dtype=np.int64)[order]
        idx_list = sorted_local.tolist()

        events = grp["events"][idx_list]
        labels = grp["labels"][idx_list]

        measurements: Optional[np.ndarray] = None
        if "measurements" in grp:
            measurements = grp["measurements"][idx_list]

        # Restore original order.
        restore = np.argsort(order)
        events = events[restore]
        labels = labels[restore]
        if measurements is not None:
            measurements = measurements[restore]

        return events, labels, measurements, layout

    # ------------------------------------------------------------------
    # Chunk loading
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[MultiRoundDataset]:
        for offset in range(0, self.total_samples, self.chunk_size):
            yield self.load_chunk(offset)

    def __len__(self) -> int:
        return self.num_chunks

    def load_chunk(
        self, offset: int, chunk_size: Optional[int] = None
    ) -> MultiRoundDataset:
        """
        Load samples starting at `offset`. Uses self.chunk_size unless chunk_size is given.

        Returns a MultiRoundDataset whose constituent SyndromeDatasets are grouped
        by HDF5 group (same basis, distance, rounds → same D), enabling
        GroupedBatchSampler to produce D-homogeneous batches.
        """
        import h5py

        size = chunk_size if chunk_size is not None else self.chunk_size
        chunk = self._sample_index[offset : offset + size]
        if len(chunk) == 0:
            raise ValueError(f"No samples at offset {offset}")

        groups: dict[int, list[int]] = {}
        for g_idx, l_idx in chunk:
            g_idx, l_idx = int(g_idx), int(l_idx)
            groups.setdefault(g_idx, []).append(l_idx)

        syndrome_datasets: List[SyndromeDataset] = []
        with h5py.File(self.h5_path, "r") as hf:
            for g_idx, local_indices in groups.items():
                events, labels, measurements, layout = self._read_group(
                    hf, g_idx, local_indices
                )
                syndrome_datasets.append(
                    SyndromeDataset(events, labels, layout, measurements)
                )

        return MultiRoundDataset(syndrome_datasets)

    def prefetching_iter(self) -> Iterator[MultiRoundDataset]:
        """
        Like __iter__ but loads the next chunk in a background thread while
        the current chunk is being trained on, hiding HDF5 I/O latency.

        Usage (replaces `for chunk in streaming_ds`):
            for chunk in streaming_ds.prefetching_iter():
                ...
        """
        offsets = list(range(0, self.total_samples, self.chunk_size))
        buf: queue.Queue = queue.Queue(maxsize=3)
        sentinel = object()

        def _worker():
            for off in offsets:
                buf.put(self.load_chunk(off))
            buf.put(sentinel)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        while True:
            item = buf.get()
            if item is sentinel:
                break
            yield item
        t.join()
