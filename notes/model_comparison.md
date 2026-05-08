# Transformer Decoder Comparison: Trans3 vs Trans5 vs Trans6 vs Trans7 vs AlphaQubit

## Overview

| Property | Trans3 | Trans5 | Trans6 | Trans7 | AlphaQubit (Paper) |
|---|---|---|---|---|---|
| **Framework** | PyTorch | PyTorch | PyTorch | PyTorch | JAX / Haiku / JAXline |
| **Basis handling** | Dual-head (X + Z) | Single-basis per model | Single-basis per model | Mixed-basis (X+Z trained jointly) | Single-basis per model |
| **Readout network** | Linear classifier | ReadoutResNet | ReadoutResNet | ReadoutResNet | ReadoutResNet (scatter -> conv -> pool -> ResNet) |
| **Embedding inputs** | 1 (syndrome value) | 1 (syndrome value) | 4 (meas, event, leak, event_leak) | 4 (meas, event, leak, event_leak) | 4 (post1, post2 + events + leakage events) |
| **Embedding ResNet** | None | None | 2-layer ResNet after summation | 2-layer ResNet after summation | 2-layer ResNet after summation |
| **Final-round handling** | No special treatment | No special treatment | Separate `final_embed` + learned off-basis vector | Separate `final_embed` + learned off-basis vector | Separate final-round embedding + off-basis embedding |
| **Data source** | Stim depolarizing noise (fresh) | Stim depolarizing noise (fresh) | Stim depolarizing noise (fresh) | Tzu-Chen fixed dataset (Stim, fixed) | SI1000 + Pauli+ (cross-talk, leakage, I/Q) + experimental |
| **Training strategy** | Single-stage | Single-stage | Single-stage | Single-stage (pretrain) | Two-stage (pretrain on simulated + finetune on experimental) |
| **Multi-GPU support** | No | No | No | Yes (DDP, torchrun) | Yes (TPU pods) |
| **LR per distance** | No (fixed 1.3e-4) | No (fixed 1.3e-4) | No (fixed 1.3e-4) | Yes (Table S3 of paper) | Yes |
| **Dilations per distance** | No (fixed) | No (fixed) | No (fixed) | Yes (Table S4 of paper) | Yes |
| **Tested distances** | d=3, d=5 | d=3, d=5, d=7 | d=3 | d=3 (in progress) | d=3, 5, 7, 9, 11 |
| **Max rounds tested** | 25 | varies | 6 | 25 | 100,000 |

---

## 1. Data Input and Structure

### 1.1 Data Generation

| Aspect | Trans3 | Trans5 | Trans6 | Trans7 | AlphaQubit |
|---|---|---|---|---|---|
| **Noise model** | Stim `surface_code:rotated_memory_{basis}` with uniform depolarization | Same as Trans3 | Same as Trans3 | Same noise model, but using the fixed Tzu-Chen dataset rather than freshly generated circuits | SI1000 (Pauli noise, non-uniform strengths), Pauli+ (cross-talk, leakage, amplitude damping), DEMs (fitted to device) |
| **Noise parameters** | `after_clifford_depolarization=p`, `before_round_data_depolarization=p`, `before_measure_flip_probability=p`, `after_reset_flip_probability=p` | Same as Trans3 | Same as Trans3 | Same as Trans3 (fixed in dataset) | SI1000: measurement noise = 5p, single-qubit/idle = p/10, 2-qubit = p. Pauli+ adds leakage channels, cross-talk unitaries |
| **Typical p values** | 0.005 | 0.005 | 0.005 | 0.005 (fixed in dataset) | ~0.001 (Pauli+ tuned for Lambda ~ 4) |
| **Shots** | 20,000–50,000 | 20,000–50,000 | 20,000 | ~1M per distance (streaming from pretrain.h5) | Up to 2.5 billion (pretrain), 10^5–10^8 (finetune) |
| **Leakage** | No | No | No (placeholder zeros) | No (placeholder zeros) | Yes, modeled in Pauli+ simulator (states \|2>, \|3>, etc.) |
| **Soft readouts (I/Q)** | No | No | No | No | Yes, 1D analogue readout with SNR and amplitude damping |
| **Cross-talk** | No | No | No | No | Yes, Pauli-twirled correlated channels on groups of up to 4 qubits |
| **Dataset mutability** | Fresh each run | Fresh each run | Fresh each run | Fixed (Tzu-Chen dataset, reproducible) | Fixed (experimental + simulator) |

### 1.2 Data Format

| Field | Trans3 | Trans5 | Trans6 | Trans7 | AlphaQubit |
|---|---|---|---|---|---|
| **Primary input** | `det_hard` or `det_soft` (N, D) | `det_hard` or `det_soft` (N, D) | `det_hard` (N, D) + `meas_hard` (N, D) | Same as Trans6, stored in `pretrain.h5` | Detection events + measurements (soft probabilities) + leakage + leakage events |
| **Labels** | `obs` (N, 2) for X and Z | `obs` (N, 1) single basis | `obs` (N, 1) single basis | `obs` (N, 1) single basis | Logical error label per basis |
| **Supplementary data** | None | None | `meas_hard` reconstructed via cumulative XOR of events per stabilizer | Same as Trans6 | Intermediate labels at every round (simulated only) |
| **Storage format** | Individual folder per (basis, distance, rounds, seed) | Same | Same | Single `pretrain.h5` HDF5 file (all 130 subdirectories compressed into 157MB) | Not specified |
| **Layout metadata** | `layout.json` with stab_id, cycle_id, x, y | Same | Same + `stab_type` (on/off basis) | Same as Trans6 (embedded in HDF5) | Per-stabilizer spatial layout, stabilizer types, circuit connectivity |

### 1.3 Input Representation to the Model

| Input channel | Trans3 | Trans5 | Trans6 | Trans7 | AlphaQubit |
|---|---|---|---|---|---|
| **Detection events** | Yes (sole input) | Yes (sole input) | Yes (via `proj_event`) | Yes (via `proj_event`) | Yes (both hard and soft) |
| **Measurements** | No | No | Yes (via `proj_meas`, cumulative XOR of events) | Yes (via `proj_meas`) | Yes (soft posterior probabilities) |
| **Leakage probability** | No | No | Placeholder zeros (`proj_leak`) | Placeholder zeros (`proj_leak`) | Yes (`post2` = posterior P(\|2>)) |
| **Leakage event** | No | No | Placeholder zeros (`proj_event_leak`) | Placeholder zeros (`proj_event_leak`) | Yes (temporal difference of leakage) |
| **Stabilizer index** | Learned embedding | Learned embedding | Learned embedding | Learned embedding | Learned embedding (+ relative positional for multi-distance) |
| **Cycle index** | Learned embedding | Learned embedding | Learned embedding | Learned embedding | Implicit via recurrent processing |

The AlphaQubit paper found that providing **both measurements and events** (rather than just events) improves performance, because measurements have a more uniform distribution and preserve asymmetry information about |0> vs |1> states (lost after XOR to events). Trans6 was the first local implementation to adopt this dual-input approach; Trans7 carries it forward.

---

## 2. Model Architecture

### 2.1 Stabilizer Embedding

| Component | Trans3 | Trans5 | Trans6 | Trans7 | AlphaQubit |
|---|---|---|---|---|---|
| **Input projections** | 1 (`analog_proj` / `val_emb`) | 1 (same) | 4 (`proj_meas`, `proj_event`, `proj_leak`, `proj_event_leak`) | 4 (same as Trans6) | 4 linear projections (Extended Data Fig. 4c) |
| **Positional info** | `stab_emb(i)` + `cycle_emb(n)` | Same | Same | Same | Stabilizer index embedding + (optional) relative positional embedding |
| **Post-summation processing** | None | None | 2-layer residual blocks (`_EmbedResBlock`) | Same as Trans6 | 2-layer ResNet (Extended Data Fig. 4c) |
| **Final-round treatment** | None | None | Separate `final_embed`; off-basis gets `offbasis_final_emb` | Same as Trans6 | Separate final-round projection; single learned vector for off-basis |

### 2.2 Syndrome Transformer (RNN Core)

The core recurrent block is structurally identical across Trans3/5/6/7. AlphaQubit uses the same design in JAX. The pseudocode is given by **Algorithm 5** (Supplementary Information, `41586_2024_8148_MOESM1_ESM.pdf`), which calls Algorithms 1-4.

```
Algorithm 5 – RNNCore (per error-correction round):
  X = (X + S_token) / sqrt(2)       # combine hidden state + new embedding
  for each layer l = 0 .. L-1:
    X = X + MHAttentionWithBias(LN(X), B)       # Alg 2 (calls Alg 1 per head)
    X = X + GatedDenseBlock(LN(X))               # Alg 3
    X = ScatteringResidualConvBlock(X)            # Alg 4 (no outer residual)
```

| Component | Trans3 | Trans5 | Trans6 | Trans7 | AlphaQubit (Pseudocode) |
|---|---|---|---|---|---|
| **Layers per round** | 3 (scaling) / 4 (internal) | 3 | 3 | 3 | 3 |
| **Multi-head attention** | Alg 2: per-head Q,K,V projections, `S=QK^T + B'`, softmax | Same | Same | Same | Same (Algorithms 1-2) |
| **Gated dense block** | Alg 3: expand to `w*d_d`, split, `GELU(a) * g`, project back | Same | Same | Same | Same (Algorithm 3) |
| **Convolution block** | Alg 4: scatter to grid -> LN -> 3x3 Conv -> GELU -> residual -> gather | Same | Same | Same | Same (Algorithm 4) |
| **Conv weight sharing** | **Shared** across all L layers | **Shared** | **Separate** per layer (`nn.ModuleList`) | **Separate** per layer | Separate per layer (standard) |
| **Dilation per distance** | Fixed (not from paper) | Fixed | Fixed | **Per-distance from Table S4** | Per-distance (Table S4) |
| **Learned padding** | Yes | Yes | Yes | Yes | Yes (learned padding vector **P**) |
| **State combination** | `(X + S) / sqrt(2)` | Same | Same | Same | Same |

### 2.3 Attention Bias

| Aspect | Trans3 | Trans5 | Trans6 | Trans7 | AlphaQubit |
|---|---|---|---|---|---|
| **Bias type** | `ManhattanDistanceBias` or `AttentionBiasProvider` | Same (full by default) | `AttentionBiasProvider` (full, default) | `AttentionBiasProvider` (full, default) | Learned attention bias embedding |
| **Features encoded** | Spatial coords + offsets + Manhattan + same-type bit + event correlations | Same | 7 features (same as Trans3 full) | 7 features (same as Trans6) | Spatial coords of i,j; signed offset; Manhattan distance; same/different basis bit |
| **Event indicator features** | 7 features in full mode | Same | 7 features | 7 features | 7 features |
| **Precomputable** | Yes (static part) | Yes | Yes | Yes | Yes |

### 2.4 Readout Network

| Component | Trans3 | Trans5 | Trans6 | Trans7 | AlphaQubit |
|---|---|---|---|---|---|
| **Architecture** | `LayerNorm -> mean pool -> Linear(d_model, 1)` | `LayerNorm -> ReadoutResNet` (global mean pool) | `LayerNorm -> ReadoutResNet` (directional pool) | Same as Trans6 | `Scatter 2D -> 2x2 Conv -> dim reduction -> mean pool perpendicular to logical observables -> ResNet -> Linear` |
| **Spatial awareness** | No | Yes (scatter 2D -> 2x2 conv) | Yes (scatter + conv, directional pool matching paper) | Same as Trans6 | Yes |
| **Residual blocks** | 0 | 16 (default) | 16 (default) | 16 (default) | 16 (Sycamore) / 4 (scaling) |
| **Hidden dim** | d_model (256) | readout_dim (48) | readout_dim (48) | readout_dim (48) | 64 (Sycamore) / 32 (scaling) |
| **Mixed-basis pooling** | N/A (dual head) | N/A (single basis) | `torch.where` selects X or Z pool direction per sample | Same as Trans6 | Separate decoders per basis |

### 2.5 Auxiliary Tasks

| Aspect | Trans3 | Trans5 | Trans6 | Trans7 | AlphaQubit |
|---|---|---|---|---|---|
| **Next-stabilizer prediction** | Yes | Yes | Yes | Yes | Yes |
| **Loss weight** | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 |
| **Weight schedule** | Constant | Cosine annealing (warmup 30% -> anneal to 0) | Constant | Cosine annealing (warmup 30% -> anneal to 0) | Not specified |
| **Intermediate labels** | Not used | Not used | Not used | Not used | Used during pretraining |

---

## 3. Parameters and Hyperparameters

### 3.1 Model Parameters

| Parameter | Trans3 (Scaling) | Trans3 (Internal) | Trans5 | Trans6 | Trans7 | AlphaQubit (Sycamore) | AlphaQubit (Scaling) |
|---|---|---|---|---|---|---|---|
| `d_model` | 256 | 256 | **128** | 256 | 256 | 320 | 256 |
| `num_heads` (H) | 4 | 8 | 4 | 4 | 4 | 4 | 4 |
| `key_size` (d_attn) | 32 | 64 | 32 | 32 | 32 | 32 | 32 |
| `d_mid` | 32 | 64 | 32 | 32 | 32 | - | - |
| `syndrome_layers` | 3 | 4 | 3 | 3 | 3 | 3 | 3 |
| `dense_widen` | 5 | 4 | 5 | 5 | 5 | 5 | 5 |
| `conv_layers` | 3 | 2 | 3 | 3 | 3 | 3 | 3 |
| `conv_dim` | 128 | 128/256 | 128 | 128 | 128 | 160 | 128 |
| `bias_dim` (db) | 48 | 32 | 48 | 48 | 48 | 48 | - (no attention bias in scaling) |
| `bias_residual_layers` | 8 | - | 8 | 8 | 8 | 8 | - |
| `readout_resnet_layers` | 16 | N/A | 16 | 16 | 16 | 16 | 4 |
| `readout_dim` | 48 | N/A | 48 | 48 | 48 | 64 | 32 |
| `feature_embed_resnet_layers` | 2 | 0 | 2 | 2 | 2 | 2 | 2 |
| **Total params (approx)** | ~5M (d=3) | varies | ~1–2M (d=3) | ~5M (d=3) | ~5M (d=3) | ~5.4M across all distances | ~5.4M across all distances |

### 3.2 Training Hyperparameters

| Hyperparameter | Trans3 | Trans5 | Trans6 | Trans7 | AlphaQubit (Sycamore) | AlphaQubit (Scaling) |
|---|---|---|---|---|---|---|
| **Optimizer** | Lion | Lion | Lion | Lion | Lamb | Lion |
| **Learning rate** | 1.3e-4 (fixed) | 1.3e-4 (fixed) | 1.3e-4 (fixed) | Per-distance (Table S3): 1.3e-4 (d=3), 1.15e-4 (d=5), 1.0e-4 (d=7), 7e-5 (d=9), 5e-5 (d=11) | 3.46e-4 (d=3), 2.45e-4 (d=5) | 7e-4 (d=3) to 3e-4 (d=11) |
| **Weight decay** | 1e-7 | 1e-7 | 1e-7 | 1e-7 | 1e-5 (pretrain), 0.08 (finetune) | 1e-7 |
| **Beta1, Beta2** | 0.9, 0.95 | 0.9, 0.95 | 0.9, 0.95 | 0.9, 0.95 | 0.9, 0.95 | 0.9, 0.95 |
| **Batch size** | 256 -> 1024 | 128 (fixed) | 256 -> 1024 | 256 (effective, 128/GPU × 2 GPUs) -> 1024 at step 800k | 256 -> 1024 (at 4M steps) | 256 -> 1024 |
| **LR schedule** | Piecewise constant (0.7x at milestones) | Piecewise constant | Piecewise constant | Piecewise constant (0.7x at milestones) | Piecewise constant | Cosine |
| **LR decay steps** | 400K, 800K, 1.6M | 400K, 800K, 1.6M | 400K, 800K, 1.6M | 400K, 800K, 1.6M | {0.8, 2, 4, 10, 20} × 10^5 | Cosine schedule |
| **Gradient clipping** | 1.0 | 1.0 | 1.0 | 1.0 | Not specified | Not specified |
| **EMA** | Yes (alpha=1e-4) | Yes (alpha=1e-4) | Yes (alpha=1e-4) | Yes (alpha=1e-4) | Yes (alpha=1e-4) | Yes |
| **Total training steps** | 2M | 500 (default) | 2M | 1M (current run) | Up to 2B samples | Up to 2.5B samples |
| **Noise curriculum** | No | No | No | No | Yes | Yes |
| **Rounds curriculum** | No | No | No | No | Not mentioned | Yes (3 -> 6 -> 12 -> 25) |
| **Next-stab weight** | 0.02 (constant) | 0.02 (cosine annealing) | 0.02 (constant) | 0.02 (cosine annealing, warmup 30%) | 0.02 | 0.01 |
| **Loss** | BCE with logits | BCE with logits | BCE with logits | BCE with logits | Cross-entropy | Cross-entropy |
| **Resume support** | No | No | No | Yes (`_resume.pth` with optimizer state) | N/A | N/A |
| **Multi-GPU** | No | No | No | Yes (DDP, torchrun, NCCL) | Yes (TPU) | Yes (TPU) |

### 3.3 Convolution Dilations

| Distance | Trans3 | Trans5 | Trans6 | Trans7 | AlphaQubit (Sycamore) | AlphaQubit (Scaling) |
|---|---|---|---|---|---|---|
| d=3 | [1, 1] or [1, 1, 1] | [1, 1, 1] | [1, 1, 1] | **[1, 1, 1]** (Table S4) | [1, 1, 1] | [1, 2, 4] |
| d=5 | [1, 1, 1] | [1, 1, 1] | - | **[1, 1, 2]** (Table S4) | [1, 1, 2] | [1, 2, 4] |
| d=7 | - | - | - | **[1, 2, 4]** (Table S4) | - | [1, 2, 4] |
| d=9 | - | - | - | **[1, 2, 4]** (Table S4) | - | [1, 2, 4] |
| d=11 | - | - | - | **[1, 2, 4]** (Table S4) | - | [1, 2, 4] |

Trans7 is the first local version to use the paper's exact dilation values per distance. All prior versions used fixed dilations that did not follow Table S4.

---

## 4. Key Architectural Differences

### 4.1 Trans3 -> Trans5 Changes
1. **Single-basis training**: Removed dual X/Z output heads. Each model trains on one basis only, matching the AlphaQubit approach.
2. **ReadoutResNet**: Replaced `Linear(d_model, 1)` with scatter-to-2D -> 2x2 conv -> mean pool -> 16 residual blocks -> Linear.
3. **Reduced d_model**: Default 128 (down from 256) as a compromise between capacity and compute.
4. **Next-stab cosine annealing**: Auxiliary loss weight anneals from 0.02 -> 0 over training.
5. **Padding strategy**: Changed from truncate-to-S_min to pad-to-S_max with boolean pad_mask.

### 4.2 Trans5 -> Trans6 Changes
1. **4-input embedding**: Four separate linear projections for measurement, event, leakage, and leakage-event inputs, summed with stabilizer/cycle embeddings, then two residual blocks.
2. **Measurement input**: Pre-computes measurements from detection events via cumulative XOR per stabilizer.
3. **Final-round handling**: Separate `final_embed` module for on-basis stabilizers in the last cycle; off-basis gets `offbasis_final_emb`.
4. **stab_type metadata**: Layout tracks on-basis vs off-basis per stabilizer.
5. **Leakage-ready architecture**: Accepts zero-valued leakage inputs for future finetuning without architecture changes.
6. **Separate conv weights per layer**: `nn.ModuleList` of independent conv blocks (vs. shared weights in Trans3/5).

### 4.3 Trans6 -> Trans7 Changes
1. **Per-distance learning rates** (from paper Table S3): d=3 → 1.3e-4, d=5 → 1.15e-4, d=7 → 1.0e-4, d=9 → 7e-5, d=11 → 5e-5. Prior versions used a fixed 1.3e-4 regardless of distance.
2. **Per-distance dilation schedules** (from paper Table S4): d=3 → [1,1,1], d=5 → [1,1,2], d≥7 → [1,2,4]. Prior versions used fixed dilations that didn't follow the paper.
3. **Multi-GPU DDP training**: `torchrun` launches DistributedDataParallel across N GPUs. Gradients are synchronized via NCCL all-reduce. Batch size is set per-GPU to keep the total effective batch at 256 regardless of GPU count.
4. **Fixed Tzu-Chen dataset**: Training uses a reproducible fixed dataset rather than freshly generated Stim circuits. Eliminates run-to-run data variation; ensures comparability across experiments.
5. **HDF5 streaming data** (`pretrain.h5`): All 130 surface code subdirectories compressed into a single 157MB file. The `ChunkedHDF5Dataset` loads data in 50k-sample chunks with background prefetching, and the file is copied to node-local `/tmp/` before training to eliminate network I/O overhead.
6. **Next-stab cosine annealing**: Restored from Trans5 (was constant in Trans6).
7. **Resume/checkpoint support**: `_resume.pth` saves optimizer momentum states, EMA state, current step, layout, and training history so jobs can be interrupted and continued exactly where they left off.
8. **Auto-requeue SLURM**: SLURM signal trap (`--signal=B:USR1@120`) automatically submits the next job 120s before the time limit, enabling seamless multi-job training runs.
9. **Mixed-basis training**: X and Z basis samples are trained together in the same run using `torch.where` for basis-conditional pooling (avoids `aten.nonzero` graph breaks in torch.compile).

### 4.4 Trans7 vs AlphaQubit (Remaining Gaps)

| Feature | Trans7 Status | AlphaQubit Paper |
|---|---|---|
| **Per-distance LR** | ✓ Implemented (Table S3) | Yes |
| **Per-distance dilations** | ✓ Implemented (Table S4) | Yes |
| **Soft I/Q readouts** | Not implemented (hard binary only) | Analogue I/Q signals with posterior probabilities |
| **Leakage simulation** | Placeholder zeros | Pauli+ model with realistic leakage channels |
| **Cross-talk** | Not modeled | Pauli-twirled correlated channels from CZ interactions |
| **Two-stage training** | Not implemented (pretrain only) | Pretrain on SI1000/DEM -> finetune on Pauli+/experimental |
| **Noise curriculum** | Not implemented | Gradually scale noise from 0.5x to 1.0x during pretrain |
| **Rounds curriculum** | Not implemented | Train on 3 -> 6 -> 12 -> 25 rounds progressively |
| **Multi-distance training** | Not implemented (one distance per run) | Single model on mixture of d=3 to d=11 |
| **Ensembling** | Not implemented | 15–20 independently trained models, averaged logits |
| **Intermediate labels** | Not used | Used during pretraining at every round |
| **Data scale** | ~1M samples (streaming) | Up to 2.5 billion pretrain + 100M finetune per distance |
| **Post-selection** | Not implemented | Confidence-based post-selection on probabilistic output |

---

## 5. Data Flow Diagrams

### Trans3 Data Flow
```
Stim circuit (depolarizing noise, single p)
  -> det_hard (N, D) binary detection events
  -> obs (N, 2) [X_label, Z_label]
  -> layout.json (stab_id, cycle_id, x, y)

Model input:
  syndrome (B, L) = flattened det_hard
  -> StabilizerEmbedding: E_stab(i) + E_cycle(n) + E_val(syndrome)
  -> RNNCore processes T cycles
  -> mean pool -> Linear -> logits_x, logits_z
```

### Trans5 Data Flow
```
Same data as Trans3, but single basis only:
  -> det_hard (N, D) binary detection events
  -> obs (N, 1) single basis label

Model input:
  syndrome (B, L) = flattened det_hard
  -> StabilizerEmbedding (same as Trans3)
  -> RNNCore processes T cycles
  -> ReadoutResNet (scatter 2D -> conv -> pool -> ResNet) -> logit
```

### Trans6 Data Flow
```
Stim circuit (depolarizing noise)
  -> det_hard (N, D) binary detection events
  -> meas_hard (N, D) = cumulative XOR of det_hard per stabilizer
  -> obs (N, 1) single basis label
  -> layout.json (stab_id, cycle_id, x, y, stab_type)

Model input per cycle t:
  meas(B,S), event(B,S), leak(B,S)=0, event_leak(B,S)=0
  -> StabilizerEmbedding:
       proj_meas(m) + proj_event(e) + proj_leak(l) + proj_event_leak(el)
       + E_stab(i) + E_cycle(n)
       -> ResBlock1 -> ResBlock2
  -> [Final cycle: on-basis uses final_embed, off-basis uses learned vector]
  -> RNNCore processes T cycles
  -> ReadoutResNet -> logit
```

### Trans7 Data Flow
```
Tzu-Chen fixed dataset (Stim, reproducible)
  -> pretrain.h5 (157MB, all distances/rounds/bases compressed)
  -> ChunkedHDF5Dataset: 50k-sample chunks, background prefetch
  -> copied to /tmp/ on compute node before training

Model input per cycle t (same as Trans6):
  meas(B,S), event(B,S), leak(B,S)=0, event_leak(B,S)=0
  -> StabilizerEmbedding (same as Trans6)
  -> [Final cycle: on-basis uses final_embed, off-basis uses learned vector]
  -> RNNCore: 3 layers, per-distance dilations from Table S4
  -> ReadoutResNet: torch.where selects X or Z pool direction per sample
  -> logit (X or Z basis depending on batch sample)

Training:
  -> DDP across N GPUs via torchrun (NCCL all-reduce)
  -> Effective batch = 256/N per GPU (total always 256, matches paper)
  -> Lion optimizer, per-distance LR from Table S3
  -> Checkpoint saved on LER improvement (_best.pth)
  -> Resume checkpoint saved every eval (_resume.pth, includes optimizer state)
  -> Auto-requeue SLURM: submits next job 120s before time limit
```

### AlphaQubit Data Flow
```
Pauli+ simulator (cross-talk, leakage, I/Q noise)
  -> I/Q readout signals -> posterior probabilities (post1, post2)
  -> Detection events (soft XOR of measurements)
  -> Leakage events (temporal diff of leakage probability)

Model input per round:
  post1 (measurement prob), events (soft), post2 (leakage prob), leakage events
  -> StabilizerEmbedder:
       Sum of 4 linear projections + E_stab(i) + E_cycle(n)
       -> 2-layer ResNet
  -> [Final round: separate final-round embedding; off-basis: learned vector]
  -> Syndrome Transformer (attention + gated dense + dilated convolutions) x3 layers
  -> Recurrently updates decoder state
  -> Readout: scatter 2D -> 2x2 conv -> project -> pool along logical observables -> ResNet -> logit
```

---

## 6. Performance Context

| Decoder | Distance 3 LER (Sycamore) | Distance 5 LER (Sycamore) | Notes |
|---|---|---|---|
| **AlphaQubit** (finetuned ensemble) | 2.901% ± 0.023% | 2.748% ± 0.015% | State of the art; Lambda = 1.056 |
| **Tensor network** | 3.028% ± 0.023% | 2.915% ± 0.016% | Best non-ML decoder, impractical at scale |
| **MWPM-BP** | 3.117% ± 0.024% | 3.059% ± 0.014% | - |
| **MWPM-Corr** | 3.498% ± 0.025% | 3.597% ± 0.015% | - |
| **PyMatching (MWPM)** | 4.015% ± 0.031% | 4.356% ± 0.019% | - |
| **Trans6** | LER = 0.0364 (d=3, r=6) | LER = 0.0249 (d=5, r=10) | Stim data, 1M shots, 50K steps; not directly comparable to Sycamore |
| **Trans7** | ~0.47 at step 15k (in progress) | Not yet evaluated | Current job at ~260k steps; loss stuck near random — likely LR or batch size issue being diagnosed |
| **Trans3/5** | Not directly comparable | - | Trained on simulated Stim data, not Sycamore experimental data |

At distance 11 on Pauli+ simulated data, AlphaQubit achieves LER approximately 5.4e-6 (ensemble), compared to MWPM-Corr at approximately 1.2e-5 (hard inputs). The model uses only 5.4M parameters across all code distances.

---

## 7. Evolution Summary

```
Trans3 (Baseline Implementation)
  |-- Faithful reproduction of core AlphaQubit components
  |-- Dual-head (X+Z) training
  |-- Simple linear readout
  |-- Manhattan distance OR full attention bias
  |-- Hard/soft syndrome input
  |
  v
Trans5 (Readout + Single-Basis)
  |-- Single-basis training (matches paper)
  |-- ReadoutResNet (spatial-aware readout, 16 residual blocks)
  |-- d_model reduced to 128 (faster training)
  |-- Cosine annealing for next-stab weight
  |-- Pad-to-S_max (preserves all detectors)
  |
  v
Trans6 (AlphaQubit Input Representation)
  |-- 4-input embedding (meas + event + leak + event_leak)
  |-- 2-layer embedding ResNet (paper Fig 4c)
  |-- Separate final-round embedding
  |-- Off-basis learned vector
  |-- Pre-computed measurements in data
  |-- stab_type (on/off basis) in layout
  |-- Leakage-ready architecture (zeros for now)
  |-- Separate conv weights per layer
  |
  v
Trans7 (Paper Hyperparameters + HPC Infrastructure)
  |-- Per-distance learning rates (Table S3)
  |-- Per-distance dilation schedules (Table S4)
  |-- Multi-GPU DDP training (torchrun, NCCL)
  |-- Fixed Tzu-Chen dataset (reproducible)
  |-- HDF5 streaming (pretrain.h5, 157MB)
  |-- Data copied to /tmp/ on compute node
  |-- Resume checkpointing (optimizer state preserved)
  |-- Auto-requeue SLURM (seamless multi-job runs)
  |-- Mixed-basis training (X+Z jointly, torch.where pooling)
  |-- Next-stab cosine annealing (restored from Trans5)
  |
  v
AlphaQubit (Full Paper, remaining gaps)
  |-- Soft I/Q readouts with posterior probabilities
  |-- Realistic leakage modeling (Pauli+ simulator)
  |-- Cross-talk modeling
  |-- Two-stage training (pretrain + finetune)
  |-- Noise and rounds curricula
  |-- Multi-distance training
  |-- Ensembling (15-20 models)
  |-- Scale: billions of training samples
```