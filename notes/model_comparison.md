# Transformer Decoder Comparison: Trans3 vs Trans5 vs Trans6 vs AlphaQubit

## Overview

| Property | Trans3 | Trans5 | Trans6 | AlphaQubit (Paper) |
|---|---|---|---|---|
| **Framework** | PyTorch | PyTorch | PyTorch | JAX / Haiku / JAXline |
| **Basis handling** | Dual-head (X + Z) | Single-basis per model | Single-basis per model | Single-basis per model |
| **Readout network** | Linear classifier | ReadoutResNet | ReadoutResNet | ReadoutResNet (scatter -> conv -> pool -> ResNet) |
| **Embedding inputs** | 1 (syndrome value) | 1 (syndrome value) | 4 (meas, event, leak, event_leak) | 4 (post1, post2 + events + leakage events) |
| **Embedding ResNet** | None | None | 2-layer ResNet after summation | 2-layer ResNet after summation |
| **Final-round handling** | No special treatment | No special treatment | Separate `final_embed` + learned off-basis vector | Separate final-round embedding + off-basis embedding |
| **Data source** | Stim circuit depolarizing noise | Stim circuit depolarizing noise | Stim circuit depolarizing noise | SI1000 + Pauli+ (cross-talk, leakage, I/Q) + experimental |
| **Training strategy** | Single-stage | Single-stage | Single-stage | Two-stage (pretrain on simulated + finetune on experimental) |
| **Tested distances** | d=3, d=5 | d=3, d=5, d=7 | d=3 | d=3, 5, 7, 9, 11 |
| **Max rounds tested** | 25 | varies | 6 | 100,000 |

---

## 1. Data Input and Structure

### 1.1 Data Generation

| Aspect | Trans3 | Trans5 | Trans6 | AlphaQubit |
|---|---|---|---|---|
| **Noise model** | Stim `surface_code:rotated_memory_{basis}` with uniform depolarization | Same as Trans3 | Same as Trans3 | SI1000 (Pauli noise, non-uniform strengths), Pauli+ (cross-talk, leakage, amplitude damping), DEMs (fitted to device) |
| **Noise parameters** | `after_clifford_depolarization=p`, `before_round_data_depolarization=p`, `before_measure_flip_probability=p`, `after_reset_flip_probability=p` | Same as Trans3 | Same as Trans3 | SI1000: measurement noise = 5p, single-qubit/idle = p/10, 2-qubit = p. Pauli+ adds leakage channels, cross-talk unitaries |
| **Typical p values** | 0.005 | 0.005 | 0.005 | ~0.001 (Pauli+ tuned for Lambda ~ 4) |
| **Shots** | 20,000-50,000 | 20,000-50,000 | 20,000 | Up to 2.5 billion (pretrain), 10^5-10^8 (finetune) |
| **Leakage** | No | No | No (placeholder zeros in embedding) | Yes, modeled in Pauli+ simulator (states |2>, |3>, etc.) |
| **Soft readouts (I/Q)** | No | No | No | Yes, 1D analogue readout with SNR and amplitude damping |
| **Cross-talk** | No | No | No | Yes, Pauli-twirled correlated channels on groups of up to 4 qubits |

### 1.2 Data Format

| Field | Trans3 | Trans5 | Trans6 | AlphaQubit |
|---|---|---|---|---|
| **Primary input** | `det_hard` or `det_soft` (N, D) | `det_hard` or `det_soft` (N, D) | `det_hard` (N, D) detection events + `meas_hard` (N, D) measurements | Detection events + measurements (soft probabilities) + leakage + leakage events |
| **Labels** | `obs` (N, 2) for X and Z | `obs` (N, 1) single basis | `obs` (N, 1) single basis | Logical error label per basis |
| **Supplementary data** | None | None | `meas_hard` reconstructed via cumulative XOR of events per stabilizer | Intermediate labels at every round (for simulated data only, not used at inference) |
| **Layout metadata** | `layout.json` with stab_id, cycle_id, x, y coordinates, num_detectors, distance | Same | Same + `stab_type` (on-basis vs off-basis per stabilizer) | Per-stabilizer spatial layout, stabilizer types, circuit connectivity |

### 1.3 Input Representation to the Model

| Input channel | Trans3 | Trans5 | Trans6 | AlphaQubit |
|---|---|---|---|---|
| **Detection events** | Yes (sole input) | Yes (sole input) | Yes (via `proj_event`) | Yes (both hard and soft) |
| **Measurements** | No (events only) | No (events only) | Yes (via `proj_meas`, cumulative XOR of events) | Yes (soft posterior probabilities, found to improve over events alone) |
| **Leakage probability** | No | No | Placeholder zeros (via `proj_leak`) | Yes (`post2` = posterior P(|2>)) |
| **Leakage event** | No | No | Placeholder zeros (via `proj_event_leak`) | Yes (temporal difference of leakage) |
| **Stabilizer index** | Learned embedding | Learned embedding | Learned embedding | Learned embedding (+ relative positional for multi-distance) |
| **Cycle index** | Learned embedding | Learned embedding | Learned embedding | Implicit via recurrent processing |

The AlphaQubit paper found that providing **both measurements and events** (rather than just events) improves performance, because measurements have a more uniform distribution and preserve asymmetry information about |0> vs |1> states (lost after XOR to events). Trans6 is the first local implementation to adopt this dual-input approach.

---

## 2. Model Architecture

### 2.1 Stabilizer Embedding

| Component | Trans3 | Trans5 | Trans6 | AlphaQubit |
|---|---|---|---|---|
| **Input projections** | 1 (`analog_proj` for soft / `val_emb` for hard) | 1 (same as Trans3) | 4 (`proj_meas`, `proj_event`, `proj_leak`, `proj_event_leak`) | 4 linear projections (Extended Data Fig. 4c) |
| **Positional info** | `stab_emb(i)` + `cycle_emb(n)` | Same | Same | Stabilizer index embedding + (optional) relative positional embedding |
| **Post-summation processing** | None (direct sum: `e_stab + e_cycle + e_val`) | None | 2-layer residual blocks (`_EmbedResBlock`) after summation | 2-layer ResNet (paper: Extended Data Fig. 4c) |
| **Final-round treatment** | None | None | Separate `final_embed` module; off-basis stabilizers get learned vector `offbasis_final_emb` | Separate final-round linear projection for on-basis; single learned embedding for off-basis |

### 2.2 Syndrome Transformer (RNN Core)

The core recurrent block is structurally identical across Trans3/5/6. AlphaQubit uses the same design in JAX. The pseudocode is given by **Algorithm 5** (Supplementary Information, `41586_2024_8148_MOESM1_ESM.pdf`), which calls Algorithms 1-4.

```
Algorithm 5 – RNNCore (per error-correction round):
  X = (X + S_token) / sqrt(2)       # combine hidden state + new embedding
  for each layer l = 0 .. L-1:
    X = X + MHAttentionWithBias(LN(X), B)       # Alg 2 (calls Alg 1 per head)
    X = X + GatedDenseBlock(LN(X))               # Alg 3
    X = ScatteringResidualConvBlock(X)            # Alg 4 (no outer residual)
```

| Component | Trans3 | Trans5 | Trans6 | AlphaQubit (Pseudocode) |
|---|---|---|---|---|
| **Layers per round** | 3 (scaling) / 4 (internal) | 3 | 3 | 3 |
| **Multi-head attention** | Alg 2: `B'=W_b B`, per-head `Q,K,V` projections, `S=QK^T + B'`, softmax. V uses `d_mid` (not `d_attn`). Output projection: `W_o(H*d_mid -> d_d)` | Same | Same | Same (Algorithms 1-2) |
| **Gated dense block** | Alg 3: `W_1` expands to `w*d_d`, split in half, `GELU(a) * g`, `W_2` projects back to `d_d` | Same | Same | Same (Algorithm 3) |
| **Convolution block** | Alg 4: scatter to `(d+1)x(d+1)` grid -> per-layer: `LN -> 3x3 Conv -> GELU -> [1x1 proj if c≠d_d]` -> residual -> gather | Same | Same | Same (Algorithm 4) |
| **Conv weight sharing** | **Shared** across all L layers (single instance) | **Shared** | **Separate** per layer (`nn.ModuleList` of L independent blocks) | Pseudocode ambiguous (separate weights per layer is standard) |
| **Learned padding** | Yes (`nn.Parameter` for empty grid cells) | Yes | Yes | Yes (learned padding vector **P**) |
| **State combination** | `(X + S) / sqrt(2)` | Same | Same | Same (scale factor `1/sqrt(2)` to control magnitude) |

### 2.3 Attention Bias

| Aspect | Trans3 | Trans5 | Trans6 | AlphaQubit |
|---|---|---|---|---|
| **Bias type** | `ManhattanDistanceBias` (simple) or `AttentionBiasProvider` (full) | Same (full by default) | `AttentionBiasProvider` (full, default) | Learned attention bias embedding (d^2-1 x d^2-1 x 48) |
| **Features encoded** | Manhattan distance between stabilizers; or spatial coords + offsets + Manhattan + same-type bit + event correlations | Same as Trans3 full | Spatial coords + offsets + Manhattan + same-type bit + event correlations (full 7-feature) | Spatial coords of i,j; signed offset; Manhattan distance; same/different basis bit |
| **Event indicator features** | 7 features (spatial and time-space event correlations) in full mode | Same | 7 features (same as Trans3 full mode) | 7 features (spatial and time-space event products) |
| **Precomputable** | Yes (static part) | Yes | Yes (static part; dynamic event features per round) | Yes (static embedding; dynamic event features require per-round update) |

### 2.4 Readout Network

| Component | Trans3 | Trans5 | Trans6 | AlphaQubit |
|---|---|---|---|---|
| **Architecture** | `LayerNorm -> mean pool over stabilizers -> Linear(d_model, 1)` | `LayerNorm -> ReadoutResNet` (global mean pool) | `LayerNorm -> ReadoutResNet` (directional pool perpendicular to logical observable) | `Scatter to 2D -> 2x2 Conv -> dim reduction -> mean pool perpendicular to logical observables -> ResNet -> Linear` |
| **Spatial awareness** | No (mean pool destroys spatial info) | Yes (scatter to 2D grid -> 2x2 conv) | Yes (scatter + conv, directional pooling matching paper) | Yes (scatter + conv, pooling perpendicular to logical observables) |
| **Residual blocks** | 0 | 16 (default) | 16 (default) | 16 (Sycamore) / 4 (scaling) |
| **Hidden dim** | d_model (256) | readout_dim (48) | readout_dim (48) | 64 (Sycamore) / 32 (scaling) |
| **Output heads** | 2 (classifier_x + classifier_z) | 1 (single basis) | 1 per basis (K position logits averaged, matching paper) | 1 per basis (d equivalent logical observable predictions averaged during training) |

### 2.5 Auxiliary Tasks

| Aspect | Trans3 | Trans5 | Trans6 | AlphaQubit |
|---|---|---|---|---|
| **Next-stabilizer prediction** | Yes (linear -> GELU -> linear per stabilizer) | Yes | Yes | Yes (linear projection + logistic output) |
| **Loss weight** | 0.02 | 0.02 (with cosine annealing schedule) | 0.02 | 0.02 (slightly detracts from final performance but faster training) |
| **Intermediate labels** | Not used | Not used | Not used | Used during pretraining (simulated data provides alternative "last rounds" at every step) |
| **Weight schedule** | Constant | Cosine annealing (warmup 30% -> anneal to 0) | Constant | Not specified (constant implied) |

---

## 3. Parameters and Hyperparameters

### 3.1 Model Parameters

| Parameter | Trans3 (Scaling) | Trans3 (Internal) | Trans5 | Trans6 | AlphaQubit (Sycamore) | AlphaQubit (Scaling) |
|---|---|---|---|---|---|---|
| `d_model` | 256 | 256 | **128** | 256 | 320 | 256 |
| `num_heads` (H) | 4 | 8 | 4 | 4 | 4 | 4 |
| `key_size` (d_attn) | 32 | 64 | 32 | 32 | 32 | 32 |
| `d_mid` | 32 | 64 | 32 | 32 | - | - |
| `syndrome_layers` | 3 | 4 | 3 | 3 | 3 | 3 |
| `dense_widen` | 5 | 4 | 5 | 5 | 5 | 5 |
| `conv_layers` | 3 | 2 | 3 | 3 | 3 | 3 |
| `conv_dim` | 128 | 128/256 | 128 | 128 | 160 | 128 |
| `bias_dim` (db) | 48 | 32 | 48 | 48 (Manhattan only) | 48 | - (no attention bias in scaling) |
| `readout_resnet_layers` | 16 | N/A (linear) | 16 | 16 | 16 | 4 |
| `readout_dim` | 48 | N/A | 48 | 48 | 64 | 32 |
| `feature_embed_resnet_layers` | 2 | 0 | 2 (from config, but not used in embedding class) | 2 (implemented in `_EmbedResBlock`) | 2 | 2 |
| **Total params (approx)** | ~5M (d=3) | varies | ~1-2M (d=3, d_model=128) | ~5M (d=3) | ~5.4M across all distances | ~5.4M across all distances |

### 3.2 Training Hyperparameters

| Hyperparameter | Trans3 | Trans5 | Trans6 | AlphaQubit (Sycamore) | AlphaQubit (Scaling) |
|---|---|---|---|---|---|
| **Optimizer** | Lion | Lion | Lion | Lamb | Lion |
| **Learning rate** | 1.3e-4 | 1.3e-4 | 1.3e-4 | 3.46e-4 (d=3), 2.45e-4 (d=5) | 7e-4 (d=3) to 3e-4 (d=11) |
| **Weight decay** | 1e-7 | 1e-7 | 1e-7 | 1e-5 (pretrain), 0.08 (finetune, relative to pretrained weights) | 1e-7 |
| **Beta1, Beta2** | 0.9, 0.95 | 0.9, 0.95 | 0.9, 0.95 | 0.9, 0.95 (b2 for Lamb) | 0.9, 0.95 |
| **Batch size** | 256 -> 1024 | 128 (fixed) | 256 -> 1024 | 256 -> 1024 (increase at 4M steps) | 256 -> 1024 |
| **LR schedule** | Piecewise constant (0.7x at milestones) | Piecewise constant | Piecewise constant | Piecewise constant (0.7x at milestones) | Cosine |
| **LR decay steps** | 400K, 800K, 1.6M | 400K, 800K, 1.6M | 400K, 800K, 1.6M | {0.8, 2, 4, 10, 20} x 10^5 | Cosine schedule |
| **Gradient clipping** | 1.0 | 1.0 | 1.0 | Not specified | Not specified |
| **EMA** | Yes (alpha=1e-4) | Yes (alpha=1e-4) | Yes (alpha=1e-4) | Yes (alpha=1e-4) | Yes |
| **Total training steps** | 2M | 500 (default, adjustable) | 2M | Up to 2B samples | Up to 2.5B samples |
| **Noise curriculum** | Not implemented | Not implemented | Not implemented | Yes (scale noise from 0.5 to 1.0 during training) | Yes |
| **Rounds curriculum** | Not implemented | Not implemented | Not implemented | Not mentioned for Sycamore | Yes (3 -> 6 -> 12 -> 25 rounds progressively) |
| **Next-stab weight** | 0.02 (constant) | 0.02 (cosine annealing) | 0.02 (constant) | 0.02 | 0.01 |
| **Loss** | BCE with logits | BCE with logits | BCE with logits | Cross-entropy | Cross-entropy |
| **pos_weight** | Auto from data | Auto from data | Auto from data | Not mentioned | Not mentioned |

### 3.3 Convolution Dilations

| Distance | Trans3 | Trans5 | Trans6 | AlphaQubit (Sycamore) | AlphaQubit (Scaling) |
|---|---|---|---|---|---|
| d=3 | [1, 1] or [1, 1, 1] | [1, 1, 1] | [1, 1, 1] | [1, 1, 1] | [1, 2, 4] |
| d=5 | [1, 1, 1] | [1, 1, 1] | - | [1, 1, 2] | [1, 2, 4] |
| d=7 | - | - | - | - | [1, 2, 4] |
| d=9 | - | - | - | - | [1, 2, 4] |
| d=11 | - | - | - | - | [1, 2, 4] |

---

## 4. Key Architectural Differences

### 4.1 Trans3 -> Trans5 Changes
1. **Single-basis training**: Removed dual X/Z output heads. Each model trains on one basis only, matching the AlphaQubit approach where decoders are basis-specific.
2. **ReadoutResNet**: Replaced `Linear(d_model, 1)` with a 6-stage pipeline: scatter to 2D -> 2x2 stride-2 conv -> 1x1 dim reduction -> global mean pool -> 16 residual blocks -> Linear(readout_dim, 1).
3. **Reduced d_model**: Default 128 (down from 256) as a compromise between capacity and compute.
4. **Next-stab cosine annealing**: Auxiliary loss weight anneals from 0.02 -> 0 over training (warmup 30% of steps at full weight, then cosine decay).
5. **Padding strategy**: Changed from truncate-to-S_min to pad-to-S_max with boolean pad_mask, preserving all detector information.

### 4.2 Trans5 -> Trans6 Changes
1. **4-input embedding**: Matches AlphaQubit Extended Data Fig. 4c. Four separate linear projections for measurement, event, leakage, and leakage-event inputs, summed with stabilizer/cycle embeddings. Two residual blocks (`_EmbedResBlock`) after summation.
2. **Measurement input**: Pre-computes measurements from detection events via cumulative XOR per stabilizer. Provides both events AND measurements to the model (paper found this improves performance).
3. **Final-round handling**: Separate `final_embed` module for on-basis stabilizers in the last cycle. Off-basis stabilizers receive a single learned parameter vector (`offbasis_final_emb`). This matches the paper's description in "Input representation" (Extended Data Fig. 4c-d).
4. **stab_type metadata**: Layout now tracks which stabilizers are on-basis vs off-basis, enabling the final-round treatment.
5. **Leakage-ready**: Although current data generation does not produce leakage, the embedding accepts zero-valued leakage inputs, allowing future finetuning with leakage data without architecture changes.
6. **Data generation**: `gen_basis_data.py` produces both `det_hard` and `meas_hard`, deterministically seeded from (base_seed, basis, distance, p).

### 4.3 Trans6 vs AlphaQubit (Remaining Gaps)

> **Note on pseudocode verification:** Algorithms 1-5 from the supplementary material (`41586_2024_8148_MOESM1_ESM.pdf`) confirm the architecture documented below. Trans6 now faithfully implements all five algorithms with per-layer conv weights, full attention bias, and directional pooling.

| Feature | Trans6 Status | AlphaQubit Paper |
|---|---|---|
| **Conv weights per layer** | ~~Shared~~ **Fixed**: separate per layer (`nn.ModuleList`) | Pseudocode ambiguous; separate per layer is standard |
| **Full attention bias** | ~~Manhattan only~~ **Fixed**: `AttentionBiasProvider` (7 features + 8-layer ResNet) is default | Learned (d^2-1)x(d^2-1)x48 embedding with spatial features + event indicators |
| **Pooling direction** | ~~Global mean pool~~ **Fixed**: directional pool perpendicular to logical observable, K logits averaged | Mean pool perpendicular to logical observables (found better than along them) |
| **Soft I/Q readouts** | Not implemented (hard binary only) | Analogue I/Q signals with posterior probabilities for |0>, |1>, |2> |
| **Leakage simulation** | Placeholder (zeros) | Pauli+ model with realistic leakage channels, transitions, cross-talk |
| **Cross-talk** | Not modeled | Pauli-twirled correlated channels from CZ gate interactions |
| **Two-stage training** | Not implemented | Pretrain on SI1000/DEM -> finetune on experimental/Pauli+ data |
| **Noise curriculum** | Not implemented | Gradually increase noise from 0.5x to 1.0x during pretraining |
| **Rounds curriculum** | Not implemented | Train on 3 -> 6 -> 12 -> 25 rounds progressively |
| **Multi-distance training** | Not implemented | Single model trained on mixture of d=3 to d=11 with relative positional embeddings |
| **Ensembling** | Not implemented | 15-20 independently trained models, averaged logits |
| **Intermediate labels** | Not used | Used during pretraining (auxiliary label at every round from simulation) |
| **Data scale** | ~20K samples | Up to 2.5 billion pretrain + 100M finetune per distance |
| **Post-selection** | Not implemented | Probabilistic output used for confidence-based post-selection |

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
| **AlphaQubit** (finetuned ensemble) | 2.901% +/- 0.023% | 2.748% +/- 0.015% | State of the art; Lambda = 1.056 |
| **Tensor network** | 3.028% +/- 0.023% | 2.915% +/- 0.016% | Best non-ML decoder, impractical at scale |
| **MWPM-BP** | 3.117% +/- 0.024% | 3.059% +/- 0.014% | - |
| **MWPM-Corr** | 3.498% +/- 0.025% | 3.597% +/- 0.015% | - |
| **PyMatching (MWPM)** | 4.015% +/- 0.031% | 4.356% +/- 0.019% | - |
| **Trans3/5/6** | Not directly comparable | - | Trained on simulated Stim data, not Sycamore experimental data |

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
  |
  v
AlphaQubit (Full Paper, remaining gaps)
  |-- Soft I/Q readouts with posterior probabilities
  |-- Realistic leakage modeling (Pauli+ simulator)
  |-- Cross-talk modeling
  |-- Two-stage training (pretrain + finetune)
  |-- Noise and rounds curricula
  |-- Full attention bias with event indicators
  |-- Pooling perpendicular to logical observables
  |-- Multi-distance training
  |-- Ensembling (15-20 models)
  |-- Scale: billions of training samples
```
