# Attention Bias Provider 使用说明

## 概述

现在提供了两种 Attention Bias Provider：

1. **ManhattanDistanceBias**（简单版本，向后兼容）
   - 只使用 Manhattan 距离
   - 适合快速测试和简单场景

2. **AttentionBiasProvider**（完整版本，符合论文）
   - 包含所有特征：Manhattan distance, coordinates, offset, type, event indicators
   - 通过 ResNet 处理特征
   - 符合 AlphaQubit 论文要求

## 使用方法

### 1. 使用完整的 AttentionBiasProvider（推荐）

```python
from trans3 import AttentionBiasProvider, AlphaQubitLikeModel
from trans3.parameter import ModelConfigScaling

# 创建 bias provider
bias_provider = AttentionBiasProvider(
    db=48,  # bias dimension (from ModelConfigScaling.bias_dim)
    max_dist=8,
    num_residual_layers=8,  # from ModelConfigScaling.bias_residual_layers
    indicator_features=7,    # from ModelConfigScaling.indicator_features
    coord_scale=0.5,  # normalize coordinates
)

# 在创建模型时使用
model = AlphaQubitLikeModel(
    num_stab=num_stab,
    num_cycles=num_cycles,
    d_model=256,
    d_attn=32,
    d_mid=64,
    db=48,
    H=4,
    n_layers=3,
    widen=4,
    conv_block=conv_block,
    bias_provider=bias_provider,  # 使用新的 bias provider
    use_next_stab=True,
)
```

### 2. 使用简单的 ManhattanDistanceBias（向后兼容）

```python
from trans3 import ManhattanDistanceBias

bias_provider = ManhattanDistanceBias(
    db=48,
    max_dist=8,
)

# 使用方式相同
model = AlphaQubitLikeModel(
    # ... 其他参数 ...
    bias_provider=bias_provider,
    # ...
)
```

## Batch 数据要求

### AttentionBiasProvider 需要的 batch 字段：

```python
batch = {
    "syndrome": torch.Tensor,      # (B, L) detection events
    "stab_xy": torch.Tensor,        # (S, 2) or (B, S, 2) stabilizer coordinates
    "stab_type": torch.Tensor,      # (S,) optional, 0=X, 1=Z
    "cycle_index": torch.Tensor,    # (T, S) optional, for event indicators
}
```

### 字段说明：

- **syndrome**: (B, L) - 检测事件值，用于构建 event indicators
- **stab_xy**: (S, 2) - 每个 stabilizer 的 (x, y) 坐标
- **stab_type**: (S,) - 可选，stabilizer 类型（0=X, 1=Z）。如果不提供，默认全部为 Z 型
- **cycle_index**: (T, S) - 可选，用于从 flat L 索引映射到 per-cycle (T, S) 索引

## 特征说明

### AttentionBiasProvider 包含的特征：

1. **Manhattan Distance** (db//4 维)
   - 两个 stabilizer 之间的 Manhattan 距离
   - 通过 Embedding 编码

2. **Coordinates** (db//4 维)
   - 每个 stabilizer 的坐标 (x, y)
   - 对每个 pair (i, j)，包含 [x_i, y_i, x_j, y_j]
   - 通过 Linear 投影

3. **Offset** (db//8 维)
   - 两个 stabilizer 之间的偏移 (dx, dy)
   - 通过 Linear 投影

4. **Type** (db//4 维)
   - Stabilizer 类型（X 或 Z）
   - 对每个 pair (i, j)，包含 [type_i, type_j]
   - 通过 Embedding 编码

5. **Event Indicators** (db//4 维)
   - 从 syndrome 值构建的事件指示器
   - 包含：event_i, event_j, event_i*event_j, |event_i-event_j|, max, min, average
   - 通过 Linear 投影

所有特征通过一个 8 层的 ResNet 处理，最终输出 (B, S, S, db) 的 bias。

## 从 ModelConfigScaling 创建

```python
from trans3.parameter import ModelConfigScaling
from trans3 import AttentionBiasProvider

model_cfg = ModelConfigScaling()

bias_provider = AttentionBiasProvider(
    db=model_cfg.bias_dim,
    num_residual_layers=model_cfg.bias_residual_layers,
    indicator_features=model_cfg.indicator_features,
)
```

## 注意事项

1. **向后兼容性**：`ManhattanDistanceBias` 仍然可用，适合简单场景
2. **性能**：`AttentionBiasProvider` 计算量更大，但提供更丰富的特征
3. **Cycle-dependent**：`AttentionBiasProvider` 支持 cycle-dependent 的 event indicators
4. **缓存**：几何特征（distance, coords, offset, type）会被缓存，因为它们不依赖于 batch 或 cycle

## 迁移指南

如果你之前使用 `ManhattanDistanceBias`，可以无缝切换到 `AttentionBiasProvider`：

```python
# 之前
from trans3.utils import ManhattanDistanceBias
bias_provider = ManhattanDistanceBias(db=48, max_dist=8)

# 现在（推荐）
from trans3 import AttentionBiasProvider
bias_provider = AttentionBiasProvider(
    db=48,
    num_residual_layers=8,
    indicator_features=7,
)
```

模型会自动检测并使用 cycle-dependent 的 bias（如果支持）。
