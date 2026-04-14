# Trans7 Model — Debugging Log

## The Problem

After implementing the trans7 AlphaQubit architecture, the model showed no learning:
- val LER hovered at **0.421–0.50** (near random chance) across a 50k-step ARC run
- Loss stayed around 0.65–0.82 throughout
- A 1M-step run showed similar/identical results

The model was correctly structured (attention, GatedDense, conv blocks, ReadoutResNet) and
gradients were flowing, but the decoder was outputting essentially random predictions.

---

## Things We Tried

### 1. Positive class weight (`pos_weight` in BCE loss)
**Hypothesis:** The label imbalance (obs_rate ≈ 0.357, i.e. ~36% errors) was biasing the model
toward predicting "no error" for everything.

**What we did:** Added auto pos_weight computation to `run_pretrain.py` (already existed in
`run_pretrain_ddp.py`). Computed as `(1 - obs_rate) / obs_rate ≈ 1.798`.

**Result:** Loss changed shape slightly (started higher, ~0.82 instead of 0.65) but val LER
still never dropped below 0.357 — the model was just predicting all-zeros with a corrected loss,
not actually learning.

---

### 2. Single-round fixed diagnostic (`run_r07_diag.slurm`)
**Hypothesis:** The curriculum (varying round counts) was making it hard to learn. Fixing to a
single round count (r=7) would simplify the problem.

**What we did:** Built `pretrain_r07.h5` from only r=7 data and ran a dedicated 50k-step training.

**Result:** Best LER was 0.357 at step 2000 (the first eval), never improved after that. Same
failure mode.

---

### 3. `torch.compile` interference
**Hypothesis:** `torch.compile` was silently breaking the forward pass in Docker (missing
Python.h headers for Triton).

**What we did:** Set `TORCHDYNAMO_DISABLE=1` before running.

**Result:** Eliminated compile-related crashes and spam, but did not affect LER.

---

### 4. Grid size bug in `build_conv_blocks` (coord_quant)
**Hypothesis:** The 2D grid used by `ScatteringResidualConvBlock` and `ReadoutResNet` was being
built from `coord_quant`-based integer rounding, which inflated a d=3 code's natural 5×5 grid
into a 17×17 or 9×9 sparse grid. Most cells were empty padding.

**Background:** `coord_quant=0.5` divides coordinates by 0.5 before rounding to integers,
doubling the effective coordinate range. The h5 files don't store `coord_quant`, so the code
used the hardcoded default of 0.5, which is wrong for AlphaQubit's integer coordinate scheme.

**Reasoning:** When using data generated from Stim, stabilizers and qubits are positioned on a grid using integers starting from 0, but are then converted into coordinates varying by 0.5 instead of just integers. Google's Surface Codes do not start from 0 nor should they be converted in 0.5's.

**What we did:** Replaced coord_quant-based indexing in `run_pretrain.py::build_conv_blocks`
with a compact unique-value mapping:

```python
# Before (broken — created 17×17 sparse grid for d=3):
q = layout.get("coord_quant", 0.5)
x_q = np.round(stab_x / q).astype(np.int32)
...
H = int(x_q.max()) + 1  # could be 17 or 9

# After (correct — creates 5×5 compact grid for d=3):
unique_x = np.unique(stab_x_all)
unique_y = np.unique(stab_y_all)
x_to_idx = {float(v): i for i, v in enumerate(unique_x)}
y_to_idx = {float(v): i for i, v in enumerate(unique_y)}
x_q = np.array([x_to_idx[float(stab_x_all[k])] for k in range(num_stab)], dtype=np.int32)
y_q = np.array([y_to_idx[float(stab_y_all[k])] for k in range(num_stab)], dtype=np.int32)
H = len(unique_x)  # 5 for d=3
W = len(unique_y)  # 5 for d=3
```

**Result:** The RNNCore conv blocks now operated on a proper 5×5 grid. Activation probe showed
`core_out` std improved. However, `logit_std` after the readout was still ~0.001–0.01 —
the model still couldn't distinguish between samples in the output. This fix was necessary but
not sufficient.

---

### 5. Overfitting diagnostic (`tiny_overfit.py`)
**What we did:** Built a dedicated script that loads 14 fixed samples and trains for 2000 steps.
Added:
- Activation variance probes (std across batch at embed, core, readout_ln, logit stages)
- Gradient norm check at step 0 (dead/no-grad parameter report)
- Per-sample logit vs label printout at the end

**Key finding from probes (before final fix):**
```
embed_out    std: 0.53   ← transformer sees different syndromes
core_out     std: 0.82   ← RNNCore differentiates samples well
readout_ln   std: 0.30   ← still distinguishable after LN
logit        std: 0.001  ← readout collapses everything to same value
```

The transformer was working correctly. The readout was discarding all per-sample information.

---

### 6. ReadoutResNet spatial_conv boundary clipping (root cause)

**Hypothesis:** The `spatial_conv` in `ReadoutResNet` was using `kernel=2, stride=2, padding=0`
on a 5×5 grid. A stride-2 conv with kernel=2 on an odd-sized grid sweeps windows starting at
columns/rows 0 and 2, covering indices 0–3. **Index 4 (the 5th row and column) is never inside
any kernel window and is silently dropped.**

For d=3, the compact grid is 5×5 (indices 0–4 in both dimensions). Stabilizers at the maximum
x or y coordinate (index 4) were scattered into the grid correctly but were then completely
ignored by the spatial_conv. These are boundary stabilizers, the ones most relevant to
detecting logical errors that cross the code boundary.

**Evidence:** Activation probes showed `readout_ln std = 0.30` (the transformer had useful
per-sample signal) but `logit std = 0.001` (the readout discarded it). The readout was
essentially averaging over the non-boundary cells and producing a nearly constant output.

**What we did:** Changed `spatial_conv` in `ReadoutResNet.__init__` from a stride-2
downsampling conv to a same-padding conv that preserves all spatial positions:

```python
# Before (broken — boundary stabilizers invisible):
self.spatial_conv = nn.Conv2d(d_model, d_model, kernel_size=2, stride=2, padding=0)
# Output for 5×5 input: 2×2 (indices 0–3 only; index 4 dropped)

# After (correct — all positions covered):
self.spatial_conv = nn.Conv2d(d_model, d_model, kernel_size=3, stride=1, padding=1)
# Output for 5×5 input: 5×5 (same-padding, all cells covered)
```

**Effect on readout positions by distance:**

| Distance | Grid  | Before (stride=2) | After (stride=1) |
|----------|-------|-------------------|-----------------|
| d=3      | 5×5   | 2 positions        | 5 positions      |
| d=5      | ~9×9  | 4 positions        | 9 positions      |
| d=7      | ~13×13| 6 positions        | 13 positions     |

**Result:** Overfit test passed. acc=1.0, loss→0.0002 within 2000 steps. Logits cleanly
separated: all positive samples at ~+8, all negative samples at -8 to -32. The model can
distinguish syndrome patterns end-to-end.

---

## Summary

The model could not learn because **the ReadoutResNet was blind to boundary stabilizers**.
A stride-2 convolution on an odd-sized (5×5) grid silently discarded the entire outermost
row and column of the grid. Since boundary stabilizers are the most informative for detecting
logical errors, the readout had almost no useful signal — producing logit_std ≈ 0.001
regardless of the input syndrome.

The `coord_quant` grid bug was also real and needed fixing (the RNNCore conv blocks were
operating on a sparse 17×17 grid), but it was masked by the readout issue.

**Both fixes applied:**
1. `run_pretrain.py::build_conv_blocks` — compact unique-value grid indexing
2. `model.py::ReadoutResNet` — `spatial_conv` changed to `kernel=3, stride=1, padding=1`
