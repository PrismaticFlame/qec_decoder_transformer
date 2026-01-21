# AlphaQubit 实现缺失功能清单

根据 `need_to_include.txt` 和代码检查，以下是还缺少的功能：

## 🔴 关键缺失功能

### 1. **Attention Bias 不完整** ⚠️ 高优先级
**当前状态：** 只有 `ManhattanDistanceBias`（仅 Manhattan 距离）
**需要添加：**
- ✅ Manhattan distance（已有）
- ❌ Coordinates (x, y)
- ❌ Offset (dx, dy)
- ❌ Type（stabilizer type）
- ❌ Event indicators（detection events）

**位置：** `utils.py` 中的 `ManhattanDistanceBias` 类需要扩展为完整的 `AttentionBiasProvider`

**参考：** 论文中的 attention bias 应该包含所有这些特征，通过一个 ResNet 处理

---

### 2. **独立的 eval.py 文件** ⚠️ 高优先级
**当前状态：** 评估功能混在 `train.py` 中
**需要：** 独立的 `eval.py` 文件，用于：
- 测试和验证（不碰 optimizer）
- 使用 EMA 模型：`ema.apply_to(model)`
- 计算不同 cycle 的 LER
- 拟合直线并计算 R²

**参考结构：**
```python
# eval.py
def evaluate_model(model, dataset, device, ema=None):
    if ema:
        ema.apply_to(model)
    model.eval()
    # ... evaluation logic
    if ema:
        ema.restore(model)
```

---

### 3. **训练入口文件 run_train.py** ⚠️ 高优先级
**当前状态：** 没有统一的入口点
**需要：** `run_train.py` 作为主入口，包含：
- 参数解析
- 数据集加载
- 模型初始化
- 调用 `train()` 函数
- Checkpoint 保存/加载

---

### 4. **完整的 Leakage Masking** ⚠️ 中优先级
**当前状态：** 只有 `mask_last_cycle` 选项
**需要实现：**
- Leakage probability：
  - `p_leak_stab = 1e-3` (0.1%)
  - `p_leak_data = 3e-3` (0.3%)
- 在数据生成时应用 leakage
- 在 dataset 中正确处理 leakage masking
- Terminal round 的特殊处理

**位置：** 
- 数据生成：`gen_soft_surrogate.py` 或 `gen_dem_si1000.py`
- Dataset：`dataset.py` 中的 masking 逻辑

---

### 5. **Terminal Round 特殊 Embedding** ⚠️ 中优先级
**当前状态：** 所有 round 使用相同的 embedding
**需要：** 最后一轮使用不同的 embedding（论文要求）

**位置：** `model.py` 中的 `StabilizerEmbedding` 类

---

### 6. **Attention Weights 调试支持** ⚠️ 低优先级
**当前状态：** `AttentionWithBiasHead.forward()` 已经返回 `A`，但没有保存到 buffer
**建议：** 添加可选的调试模式，保存 attention weights 到 buffer：
```python
def forward(self, X, Bp, debug=False):
    # ... existing code ...
    if debug:
        self.register_buffer('last_attn', A)
    return out, A
```

---

## ✅ 已实现功能

1. ✅ **模型架构**：RNNCore, AttentionWithBias, GatedDenseBlock, ScatteringResidualConvBlock
2. ✅ **分别的 X/Z 标签支持**：模型和 loss 都支持
3. ✅ **Soft input pipeline**：`analog_proj` 在 `StabilizerEmbedding` 中
4. ✅ **EMA**：`ema.py` 已实现
5. ✅ **Lion optimizer**：`utils.py` 中已实现
6. ✅ **Fit gate utilities**：`fit_line_with_stats`, `gate_fit_ok` 已实现
7. ✅ **Layout generation**：`layout.py` 已实现
8. ✅ **数据生成**：`gen_soft_surrogate.py`, `gen_dem_si1000.py` 已实现
9. ✅ **训练循环**：`train.py` 基本完整
10. ✅ **Loss 函数**：`loss.py` 支持 X/Z 分别和 next-stab

---

## 📋 文件结构对比

### 当前结构：
```
trans3(based on alphaqubit)/
├── model.py          ✅
├── parameter.py      ✅
├── dataset.py        ✅
├── train.py          ✅ (但评估混在里面)
├── loss.py           ✅
├── utils.py          ✅
├── ema.py            ✅
├── layout.py         ✅
├── gen_soft_surrogate.py  ✅
├── gen_dem_si1000.py      ✅
├── visualization.py        ✅
└── __init__.py            ✅
```

### 期望结构（根据 need_to_include.txt）：
```
trans3(based on alphaqubit)/
├── model.py          ✅
├── parameter.py      ✅
├── dataset.py        ✅
├── train.py          ✅
├── eval.py           ❌ 缺失
├── run_train.py      ❌ 缺失
├── loss.py           ✅
├── utils.py          ⚠️ 需要扩展 AttentionBias
├── ema.py            ✅
├── layout.py         ✅
├── gen_soft_surrogate.py  ✅
├── gen_dem_si1000.py      ✅
└── ...
```

---

## 🎯 优先级建议

1. **立即处理：**
   - 创建 `eval.py`
   - 创建 `run_train.py`
   - 扩展 Attention Bias（coords/offset/type/event indicators）

2. **短期处理：**
   - 实现完整的 leakage masking
   - Terminal round 特殊 embedding

3. **可选优化：**
   - Attention weights 调试支持
   - 更多可视化工具

---

## 📝 注意事项

- **Attention bias** 是论文的核心特性，需要完整实现
- **eval.py** 应该独立于训练循环，便于单独运行测试
- **Leakage masking** 对模型性能有重要影响，需要正确实现
- 所有功能都应该支持 **25 cycles**（论文要求）
