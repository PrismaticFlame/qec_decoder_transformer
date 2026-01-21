# Evaluation Module 使用说明

## 概述

`eval.py` 提供了独立的评估功能，不涉及 optimizer，专门用于模型评估和测试。

## 主要功能

### 1. `evaluate_model` - 主评估函数

最常用的评估函数，支持两种模式：

```python
from trans3 import evaluate_model, EMA
from torch.utils.data import DataLoader

# 使用 EMA 模型评估
metrics = evaluate_model(
    model=model,
    val_loader=val_loader,
    device=device,
    ema=ema,              # EMA 对象（可选）
    use_ema=True,         # 是否使用 EMA（默认 True）
    cycles=[3, 5, 7, 9, 11, 13, 15, 25],
    eval_mode="sycamore", # "sycamore" 或 "simple"
    min_r2=0.9,
    min_intercept=-0.02,
    return_xz=False,     # 是否返回 X/Z 分别的 LER
)

# 返回的 metrics 包含：
# - dev/ler_pred25: 预测的 cycle 25 LER
# - dev/fit_r2: 线性拟合的 R²
# - dev/fit_slope: 斜率
# - dev/fit_intercept: 截距
# - dev/fit_ok: 拟合是否通过质量门控
# - dev/ler_cycle_{c}: 每个 cycle 的 LER
```

### 2. `evaluate_test_set` - 测试集评估

专门用于测试集评估，使用 "test/" 前缀：

```python
from trans3 import evaluate_test_set

test_metrics = evaluate_test_set(
    model=model,
    test_loader=test_loader,
    device=device,
    ema=ema,
    use_ema=True,
    cycles=[3, 5, 7, 9, 11, 13, 15, 25],
    return_xz=False,
)

# 返回的 metrics 使用 "test/" 前缀：
# - test/ler_pred25
# - test/fit_r2
# - test/ler_cycle_{c}
# 等等
```

### 3. `evaluate_ler_with_fit` - 带拟合的评估

Sycamore 风格的评估，包含线性拟合：

```python
from trans3 import evaluate_ler_with_fit

metrics = evaluate_ler_with_fit(
    model=model,
    val_loader=val_loader,
    device=device,
    ema=ema,
    cycles_for_fit=[3, 5, 7, 9, 11, 13, 15, 25],
    min_r2=0.9,
    min_intercept=-0.02,
    use_ema=True,
    return_xz=False,
)
```

### 4. `compute_cycle_ler` - 计算特定 cycle 的 LER

直接计算指定 cycles 的 LER：

```python
from trans3 import compute_cycle_ler

ler_by_cycle = compute_cycle_ler(
    model=model,
    val_loader=val_loader,
    device=device,
    cycles=[3, 5, 7, 25],
    use_ema=True,
    ema=ema,
    return_xz=False,
)

# 返回: {3: ler_3, 5: ler_5, 7: ler_7, 25: ler_25}
```

## EMA 使用

所有评估函数都支持 EMA 模型：

```python
from trans3 import EMA, evaluate_model

# 创建 EMA（在训练时）
ema = EMA(model, alpha=1e-4)

# 训练循环中更新 EMA
for batch in train_loader:
    # ... forward, backward, optimizer.step() ...
    ema.update(model)  # 在 optimizer.step() 之后

# 评估时使用 EMA
metrics = evaluate_model(
    model=model,
    val_loader=val_loader,
    device=device,
    ema=ema,
    use_ema=True,  # 使用 EMA 模型
)
```

**重要：** `eval.py` 中的函数会自动处理 EMA 的 apply 和 restore，确保：
- 评估时使用 EMA shadow weights
- 评估后恢复原始 weights（不影响训练）

## X/Z 分别评估

支持分别评估 X 和 Z 型逻辑错误：

```python
metrics = evaluate_model(
    model=model,
    val_loader=val_loader,
    device=device,
    ema=ema,
    return_xz=True,  # 启用 X/Z 分别评估
)

# 返回额外的 metrics：
# - dev/ler_x: X 型逻辑错误率
# - dev/ler_z: Z 型逻辑错误率
# - dev/ler_x_cycle_{c}: 每个 cycle 的 X LER
# - dev/ler_z_cycle_{c}: 每个 cycle 的 Z LER
```

## 评估模式

### Sycamore 模式（默认）

使用线性拟合预测 cycle 25 的 LER：

```python
metrics = evaluate_model(
    model=model,
    val_loader=val_loader,
    device=device,
    eval_mode="sycamore",  # 默认
    cycles=[3, 5, 7, 9, 11, 13, 15, 25],
    min_r2=0.9,           # 最小 R²
    min_intercept=-0.02,  # 最小截距
)
```

### Simple 模式

只计算 cycle 25 的 LER，不进行拟合：

```python
metrics = evaluate_model(
    model=model,
    val_loader=val_loader,
    device=device,
    eval_mode="simple",
)
```

## 完整示例

```python
import torch
from torch.utils.data import DataLoader
from trans3 import (
    AlphaQubitLikeModel,
    SyndromeDataset,
    EMA,
    evaluate_model,
    evaluate_test_set,
)

# 加载模型
model = AlphaQubitLikeModel(...)
model.load_state_dict(torch.load("checkpoint.pth"))

# 创建 EMA（如果 checkpoint 中有 EMA state）
ema = EMA(model, alpha=1e-4)
ema.load_state_dict(torch.load("ema_checkpoint.pth"))

# 加载数据集
val_dataset = SyndromeDataset(...)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

test_dataset = SyndromeDataset(...)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 验证集评估
val_metrics = evaluate_model(
    model=model,
    val_loader=val_loader,
    device=device,
    ema=ema,
    use_ema=True,
    eval_mode="sycamore",
    return_xz=True,
)

print(f"Validation LER@25: {val_metrics['dev/ler_pred25']:.6f}")
print(f"Validation LER_X: {val_metrics['dev/ler_x']:.6f}")
print(f"Validation LER_Z: {val_metrics['dev/ler_z']:.6f}")

# 测试集评估
test_metrics = evaluate_test_set(
    model=model,
    test_loader=test_loader,
    device=device,
    ema=ema,
    use_ema=True,
    return_xz=True,
)

print(f"Test LER@25: {test_metrics['test/ler_pred25']:.6f}")
print(f"Test LER_X: {test_metrics['test/ler_x']:.6f}")
print(f"Test LER_Z: {test_metrics['test/ler_z']:.6f}")
```

## 注意事项

1. **不涉及 optimizer**：`eval.py` 中的所有函数都是纯评估，不会修改模型参数（除了临时应用 EMA）
2. **自动 EMA 管理**：函数会自动处理 EMA 的 apply 和 restore，确保不影响训练状态
3. **向后兼容**：支持旧版本的模型输出格式（logical_logits, logits, logit）
4. **X/Z 支持**：如果模型输出分别的 X/Z logits，可以启用 `return_xz=True` 获取分别的评估结果
