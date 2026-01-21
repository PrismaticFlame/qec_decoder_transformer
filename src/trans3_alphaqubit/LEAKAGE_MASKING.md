# Leakage Masking 使用说明

## 概述

Leakage masking 模拟了实际量子硬件中测量丢失的情况。在 QEC 中，某些 stabilizer 或 data qubit 的测量可能会因为各种原因（如 readout 失败、时序问题等）而丢失。

## Leakage 概率

根据论文和实际硬件特性：
- **p_leak_stab = 1e-3** (0.1%): Stabilizer detectors 的 leakage 概率
  - 适用于中间 cycles 的 ancilla qubit 测量
- **p_leak_data = 3e-3** (0.3%): Data qubit detectors 的 leakage 概率
  - 适用于最后一轮的 data qubit 测量
  - 通常比 stabilizer 更高，因为 data qubit 测量更复杂

## 实现位置

### 1. 数据生成 (`gen_soft_surrogate.py`)

在生成数据时应用 leakage：

```python
from gen_soft_surrogate import gen_soft_surrogate_dataset

circ, det_hard, det_soft, obs, leakage_mask = gen_soft_surrogate_dataset(
    distance=3,
    rounds=5,
    p=1e-3,
    shots=20000,
    mu=1.2,
    sigma=1.0,
    seed_meas=42,
    seed_analog=42,
    apply_leakage=True,      # 启用 leakage
    p_leak_stab=1e-3,        # Stabilizer leakage 概率
    p_leak_data=3e-3,        # Data qubit leakage 概率
    seed_leakage=42,         # Leakage 随机种子
)

# 保存数据时包含 leakage_mask
np.savez_compressed(
    "data.npz",
    det_hard=det_hard,
    det_soft=det_soft,
    obs=obs,
    leakage_mask=leakage_mask.astype(np.uint8),  # 保存为 uint8 节省空间
)
```

### 2. Dataset 使用 (`dataset.py`)

Dataset 会自动加载和使用 leakage mask：

```python
from dataset import SyndromeDataset

# 如果 .npz 文件包含 leakage_mask，会自动加载
dataset = SyndromeDataset(
    samples=samples,
    labels=labels,
    layout_json_path="layout.json",
    input_mode="soft",
    leakage_mask=leakage_mask,  # 可选：也可以直接传入
)

# Dataset 会自动：
# 1. 在 __getitem__ 中应用 leakage mask 到 syndrome
# 2. 在 token_mask 中应用 leakage mask（用于 next-stab loss）
```

### 3. 训练使用 (`run_train.py`)

训练时会自动使用 leakage mask：

```bash
python run_train.py \
    --train_data data/train.npz \
    --val_data data/val.npz \
    --layout data/layout.json
```

如果 .npz 文件包含 `leakage_mask`，会自动加载并使用。

## Leakage Mask 格式

### 生成时的格式

`leakage_mask` 是一个 `(N, D)` 的 bool 数组：
- `True`: Detector 被保留（正常测量）
- `False`: Detector 泄漏（测量丢失）

### 保存格式

保存为 `uint8` 以节省空间：
- `1` = 保留
- `0` = 泄漏

### 在 Dataset 中的处理

1. **Syndrome masking**: Leaked detectors 的值被设为 0（hard）或 0.0（soft）
2. **Token mask**: Leaked detectors 在 `token_mask` 中被标记为 `False`，在 next-stab loss 中被忽略

## 工作原理

### 1. Detector 类型判断

根据 `cycle_id` 判断 detector 类型：
- **Stabilizer detectors**: `cycle_id < max_cycle` → 使用 `p_leak_stab`
- **Data qubit detectors**: `cycle_id == max_cycle` → 使用 `p_leak_data`

### 2. Leakage 采样

对每个 detector 和每个 shot，独立采样 leakage：
```python
for each detector d:
    if cycle_id[d] == max_cycle:
        p_leak = p_leak_data  # 0.3%
    else:
        p_leak = p_leak_stab  # 0.1%
    
    for each shot n:
        if random() < p_leak:
            leakage_mask[n, d] = False  # Leaked
        else:
            leakage_mask[n, d] = True   # Kept
```

### 3. 应用 Leakage

在数据生成时：
- Hard detectors: `det_hard[leaked] = 0`
- Soft detectors: `det_soft[leaked] = 0.0` (neutral LLR)

在训练时：
- Syndrome input: 自动应用 leakage mask
- Next-stab loss: 通过 `token_mask` 忽略 leaked detectors

## 完整示例

### 生成带 Leakage 的数据

```python
import numpy as np
from gen_soft_surrogate import gen_soft_surrogate_dataset

# 生成数据（带 leakage）
circ, det_hard, det_soft, obs, leakage_mask = gen_soft_surrogate_dataset(
    distance=3,
    rounds=25,  # 25 cycles as per paper
    p=1e-3,
    shots=20000,
    mu=1.2,
    sigma=1.0,
    apply_leakage=True,
    p_leak_stab=1e-3,
    p_leak_data=3e-3,
    seed_leakage=42,
)

# 保存
np.savez_compressed(
    "data/train_with_leakage.npz",
    det_hard=det_hard,
    det_soft=det_soft,
    obs=obs,
    leakage_mask=leakage_mask.astype(np.uint8),
)

print(f"Leakage rate: {(1 - leakage_mask.mean()) * 100:.2f}%")
```

### 使用带 Leakage 的数据训练

```python
from dataset import SyndromeDataset

# 加载数据（自动检测 leakage_mask）
train_data = np.load("data/train_with_leakage.npz")
dataset = SyndromeDataset(
    samples=train_data["det_soft"],
    labels=train_data["obs"],
    layout_json_path="data/layout.json",
    input_mode="soft",
)

# leakage_mask 会自动应用
batch = dataset[0]
# batch["syndrome"] 已经应用了 leakage mask
# batch["token_mask"] 已经包含了 leakage information
```

## 验证 Leakage

### 检查 Leakage 率

```python
import numpy as np

data = np.load("data/train_with_leakage.npz")
leakage_mask = data["leakage_mask"].astype(bool)

# 总体 leakage 率
overall_rate = 1.0 - leakage_mask.mean()
print(f"Overall leakage rate: {overall_rate:.4%}")

# 按 cycle 统计
# (需要 layout.json 中的 cycle_id)
```

### 可视化 Leakage 模式

```python
import matplotlib.pyplot as plt

leakage_mask = data["leakage_mask"]

# 可视化某个 shot 的 leakage 模式
plt.imshow(leakage_mask[0].reshape(num_cycles, num_stab), cmap='gray')
plt.title("Leakage Pattern (white=kept, black=leaked)")
plt.xlabel("Stabilizer")
plt.ylabel("Cycle")
plt.show()
```

## 注意事项

1. **Terminal Round**: 最后一轮通常包含 data qubit detectors，leakage 率更高
2. **Per-shot Leakage**: 每个 shot 的 leakage 模式是独立的（更真实）
3. **Mask Combination**: Leakage mask 会与 `mask_last_cycle` 和 `custom_token_mask` 组合
4. **Loss Masking**: Leakage 的 detectors 在 next-stab loss 中会被自动忽略

## 与 Terminal Round 的关系

Terminal round（最后一轮）的特殊处理：
- 最后一轮通常包含 data qubit measurements
- Data qubit detectors 有更高的 leakage 率（3e-3 vs 1e-3）
- 可以使用 `mask_last_cycle=True` 完全 mask 最后一轮（避免 label leakage）
- 或者使用 leakage mask 部分 mask（更真实）

## 向后兼容

- 如果数据文件不包含 `leakage_mask`，dataset 会正常工作（所有 detectors 都被保留）
- 如果 `apply_leakage=False`，数据生成不会应用 leakage
- 旧的代码仍然可以工作（leakage mask 是可选的）
