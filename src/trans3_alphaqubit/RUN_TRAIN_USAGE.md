# run_train.py 使用说明

## 概述

`run_train.py` 是训练 AlphaQubit-like QEC Transformer 的统一入口点，提供了完整的训练流程，包括：
- 数据加载
- 模型构建
- 训练执行
- Checkpoint 保存/加载

## 基本用法

### 1. 最简单的用法

```bash
python run_train.py \
    --train_data data/train.npz \
    --val_data data/val.npz \
    --layout data/layout.json
```

### 2. 使用配置文件

```bash
python run_train.py \
    --config config.json \
    --train_data data/train.npz \
    --val_data data/val.npz \
    --layout data/layout.json
```

配置文件格式 (`config.json`):
```json
{
    "model": {
        "d_model": 256,
        "num_heads": 4,
        "key_size": 32,
        "syndrome_layers": 3,
        "conv_layers": 3,
        "conv_dim": 128,
        "bias_dim": 48,
        "bias_residual_layers": 8,
        "indicator_features": 7
    },
    "training": {
        "num_steps": 2000000,
        "lr": 1.3e-4,
        "batch_init": 256,
        "batch_final": 1024,
        "use_ema": true,
        "ema_alpha": 1e-4
    }
}
```

### 3. 从 checkpoint 恢复训练

```bash
python run_train.py \
    --train_data data/train.npz \
    --val_data data/val.npz \
    --layout data/layout.json \
    --resume checkpoints/scaling_run_final.pth
```

## 命令行参数

### 必需参数

- `--train_data`: 训练数据路径 (.npz 文件)
- `--val_data`: 验证数据路径 (.npz 文件)
- `--layout`: Layout JSON 文件路径

### 可选参数

#### 模型选项
- `--use_full_bias`: 使用完整的 AttentionBiasProvider（默认），否则使用 ManhattanDistanceBias
- `--input_mode`: 输入模式，`hard` 或 `soft`（默认：`soft`）
- `--mask_last_cycle`: 是否 mask 最后一轮（避免 label leakage）

#### 训练选项
- `--config`: 配置文件路径
- `--resume`: Checkpoint 路径（恢复训练）
- `--run_name`: 运行名称（用于 wandb/logging，默认：`scaling_run`）
- `--use_wandb`: 是否使用 wandb（默认：True）
- `--output_dir`: Checkpoint 保存目录（默认：`./checkpoints`）

#### 覆盖配置选项
- `--num_steps`: 训练步数
- `--batch_init`: 初始 batch size
- `--lr`: 学习率
- `--seed`: 随机种子

## 数据格式

### .npz 文件格式

训练/验证数据文件应包含以下键：

```python
{
    "det_soft": np.ndarray,  # (N, L) soft detector values (推荐)
    "det_hard": np.ndarray,  # (N, L) hard detector values (可选)
    "obs": np.ndarray,       # (N, 2) 或 (N, 1) logical labels
                             # 如果是 (N, 2): [X, Z] labels
                             # 如果是 (N, 1): 单一 label（向后兼容）
}
```

或者使用旧格式：
```python
{
    "samples": np.ndarray,   # (N, L) detector values
    "labels": np.ndarray,    # (N, 2) 或 (N, 1) logical labels
}
```

### layout.json 格式

Layout 文件应包含：
```json
{
    "num_detectors": int,
    "num_stab": int,
    "num_cycles": int,
    "distance": int,
    "stab_id": [int, ...],
    "cycle_id": [int, ...],
    "x": [float, ...],
    "y": [float, ...],
    ...
}
```

## 完整示例

### 示例 1: 基本训练

```bash
python run_train.py \
    --train_data data/train.npz \
    --val_data data/val.npz \
    --layout data/layout.json \
    --run_name my_experiment \
    --num_steps 1000000 \
    --batch_init 512 \
    --lr 1e-4
```

### 示例 2: 使用完整 Attention Bias

```bash
python run_train.py \
    --train_data data/train.npz \
    --val_data data/val.npz \
    --layout data/layout.json \
    --use_full_bias \
    --input_mode soft \
    --run_name full_bias_experiment
```

### 示例 3: 从 Checkpoint 恢复

```bash
python run_train.py \
    --train_data data/train.npz \
    --val_data data/val.npz \
    --layout data/layout.json \
    --resume checkpoints/scaling_run_final.pth \
    --run_name resumed_training \
    --num_steps 2000000
```

### 示例 4: 使用配置文件

```bash
# 创建 config.json
cat > config.json << EOF
{
    "model": {
        "d_model": 256,
        "num_heads": 4,
        "syndrome_layers": 3
    },
    "training": {
        "num_steps": 2000000,
        "lr": 1.3e-4,
        "batch_init": 256
    }
}
EOF

# 运行训练
python run_train.py \
    --config config.json \
    --train_data data/train.npz \
    --val_data data/val.npz \
    --layout data/layout.json
```

## Checkpoint 格式

保存的 checkpoint 包含：
```python
{
    "model_state_dict": {...},      # 模型参数
    "optimizer_state_dict": {...},  # Optimizer 状态（如果提供）
    "step": int,                    # 当前步数
    "best_ler": float,              # 最佳 LER
    "fit_r2": float,                # 拟合 R²（可选）
    "fit_intercept": float,        # 拟合截距（可选）
}
```

## 输出

训练完成后，会在 `--output_dir` 目录下保存：
- `{run_name}_final.pth`: 最终 checkpoint

训练过程中，最佳模型会被保存在内存中，并在训练结束时保存。

## 注意事项

1. **Layout 文件必须包含 `distance` 字段**，用于构建 conv_block
2. **数据文件应包含 `det_soft` 或 `samples` 键**，以及 `obs` 或 `labels` 键
3. **如果使用 `--resume`，模型架构必须匹配** checkpoint 中的模型
4. **wandb 需要正确配置**（如果使用 `--use_wandb`）

## 故障排除

### 问题：找不到模块
**解决**：确保在 `src/trans3(based on alphaqubit)/` 目录下运行，或使用正确的 Python 路径

### 问题：Layout 缺少 distance
**解决**：确保 layout.json 包含 `distance` 字段，或让脚本自动推断

### 问题：数据格式不匹配
**解决**：检查 .npz 文件是否包含正确的键（`det_soft`, `obs` 等）

### 问题：CUDA out of memory
**解决**：减小 `--batch_init` 或使用 CPU（`--device cpu` 在配置中）
