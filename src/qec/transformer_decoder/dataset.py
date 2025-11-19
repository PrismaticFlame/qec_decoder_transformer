import numpy as np
import torch
from torch.utils.data import Dataset


class SyndromeDataset(Dataset):
    """
    Dataset for QEC transformer (hard-input AlphaQubit style).

    detector_samples: (N_shots, N_detectors), values in {0, 1}
    logical_labels:   (N_shots, 1) or (N_shots,), values in {0, 1}
    rounds:           number of measurement rounds used in Stim circuit
    """

    def __init__(self, detector_samples: np.ndarray, logical_labels: np.ndarray, rounds: int):
        # to numpy
        detector_samples = np.asarray(detector_samples)
        logical_labels = np.asarray(logical_labels)

        # ensure 2D
        if detector_samples.ndim == 1:
            detector_samples = detector_samples.reshape(1, -1)
        if logical_labels.ndim == 1:
            logical_labels = logical_labels.reshape(-1, 1)

        assert detector_samples.shape[0] == logical_labels.shape[0], \
            "Number of shots must match between detector_samples and logical_labels"

        self.dets = torch.from_numpy(detector_samples).long()   # (N, L)
        self.labels = torch.from_numpy(logical_labels).float()  # (N, 1) for BCEWithLogitsLoss

        # ---- detector / stabilizer layout ----
        self.num_shots, self.num_detectors = self.dets.shape
        self.num_cycles = rounds

        assert self.num_detectors % self.num_cycles == 0, \
            f"num_detectors={self.num_detectors} not divisible by rounds={self.num_cycles}"

        self.num_stab_per_round = self.num_detectors // self.num_cycles

        # map detector index j -> (stab_id, cycle_id)
        stab_id = np.array(
            [j % self.num_stab_per_round for j in range(self.num_detectors)],
            dtype=np.int64,
        )
        cycle_id = np.array(
            [j // self.num_stab_per_round for j in range(self.num_detectors)],
            dtype=np.int64,
        )

        self.stab_id = torch.from_numpy(stab_id)    # (L,)
        self.cycle_id = torch.from_numpy(cycle_id)  # (L,)

    def __len__(self):
        return self.num_shots

    def __getitem__(self, idx):
        """
        Returns:
            {
                "syndrome": (L,), 0/1
                "label":    (1,), float 0/1
                "stab_id":  (L,),
                "cycle_id": (L,),
            }
        """
        x = self.dets[idx]      # (L,)
        y = self.labels[idx]    # (1,)

        return {
            "syndrome": x,
            "label": y,
            "stab_id": self.stab_id,
            "cycle_id": self.cycle_id,
        }
