# Optimizations for Transformer 7

## Implemented

| Optimization | Location | Effective? | Notes |
| --- | --- | --- | --- |
| Mixed Precision (AMP) | train.py | Yes | bfloat16 on H200 (no GradScaler needed); float16 + GradScaler fallback on older GPUs |
| TF32 matmul | run_pretrain_ddp.py | Yes | `torch.set_float32_matmul_precision('high')` — up to ~1.7x on matmul/conv ops on Ampere+ |
| Multi-GPU (DDP) | run_pretrain_ddp.py | Yes | 2x H200s; linear scaling with GPU count for compute-bound training |
| Flash Attention (SDPA) | model.py | Yes | `F.scaled_dot_product_attention` — PyTorch selects fused FlashAttention kernel automatically |
| int8 data storage | dataset.py | Yes | Events/measurements stored as int8, cast to float32 in `__getitem__`; 8x RAM reduction vs int64 |
| DataLoader prefetch + persistent workers | dataset.py | Yes | `prefetch_factor=4`, `persistent_workers=True` — reduces CPU↔GPU pipeline stalls |
| Joint input projection | model.py | Yes | Single `Linear(2, d_model)` replaces two `Linear(1, d_model)` in StabilizerEmbedding and FinalRoundEmbedding; better GEMM utilization |
| Operator Fusion (torch.compile) | run_pretrain.py, run_pretrain_ddp.py | Potentially | Currently compiling `model.core` + `model.bias_provider` individually to avoid O(T) recompilation |
| Split AttentionBiasProvider (geometry cache) | utils.py, hyperparameters.py | Unknown | Geometry ResNet runs once on (S,S) with no batch dim, cached during eval; interaction ResNet runs per step. Reduces bias provider FLOPs at cost of some expressivity |
| Gradient Accumulation | train.py, hyperparameters.py | No | Implemented but set to 1 (disabled); DDP already gives large effective batch; overhead not justified |
| Gradient Checkpointing | model.py | Disabled | Reduces activation memory at ~30% compute cost; disabled because Dynamo's `lift_tracked_freevar` assertion fires when used with `torch.compile` |
| Cached `_bias_takes_cycle` flag | model.py | Minor | Avoids `inspect.signature` call in the forward loop per cycle step |

## Not Yet Tried

| Optimization | Expected Benefit | Blocker / Notes |
| --- | --- | --- |
| Dropout | Regularization — may help or hurt | Unknown for QEC; AlphaQubit paper does not use it |
| `channels_last` memory format | Better Conv2d throughput on NVIDIA (avoids NCHW→NHWC transpose) | Requires `.to(memory_format=torch.channels_last)` on model + input tensors |

## Compile Issues Encountered

Compiling the full outer model (`torch.compile(model)`) with DDP caused two bugs:

1. **O(T) recompilation**: The Python `for t in range(T)` (`T` is shots/cycles) loop gets unrolled into a single graph per unique T value. With T up to 26 (r=25 curriculum), each new round count triggers a full recompile. Exceeds the NCCL 10-minute all-reduce timeout and hits OOM during compilation.
   - **Fix**: Compile `model.core` and `model.bias_provider` directly. Each is compiled once for its fixed input shape, independent of T.

2. **`BackendCompilerFailed: 'float' object has no attribute 'meta'`**: DDPOptimizer splits the compiled graph at gradient bucket boundaries; Python float module attributes (e.g. `coord_scale`) become submodule outputs in the split subgraphs and fail Inductor's `meta` check.
   - **Fix**: `torch._dynamo.config.optimize_ddp = False`

3. **`AssertionError: lift_tracked_freevar_to_input`**: `grad_checkpoint` uses a lambda that captures `self` as a closure variable; Dynamo can't lift it to a proper subgraph input.
   - **Fix**: `use_grad_checkpoint=False` when compiling.
