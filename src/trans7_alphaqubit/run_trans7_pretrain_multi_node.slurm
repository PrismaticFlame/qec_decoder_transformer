#!/bin/bash
######################## SLURM OPTIONS #########################################
#SBATCH --job-name=trans7_multi
#SBATCH --account=quantum_training
#SBATCH --time=168:00:00
#SBATCH --partition=h200_normal_q
#SBATCH --nodes=2                   # number of nodes  ──┐ edit both together
#SBATCH --gres=gpu:4                # GPUs per node    ──┘ (nodes × gres = total GPUs)
#SBATCH --ntasks-per-node=1         # one torchrun launcher per node
#SBATCH --cpus-per-task=32          # CPUs shared across all GPU workers on this node
#SBATCH --mem=128G
#SBATCH --output=logs/trans7_multi_%j.out
#SBATCH --error=logs/trans7_multi_%j.err
#SBATCH --signal=B:USR1@120
###############################################################################
# Trans7 AlphaQubit Pretraining — Multi-Node DDP
#
# Splits GPUs across multiple nodes to improve queue time at the cost of
# slightly higher inter-node AllReduce overhead (InfiniBand vs NVLink).
#
# ARC has 7 nodes of 8 H200 GPUs each. Recommended configs:
#   --nodes=2  --gres=gpu:4   (8 GPUs total, 2 InfiniBand hops)
#   --nodes=4  --gres=gpu:2   (8 GPUs total, easier to schedule, ~10-15% slower)
#
# Note: --gres and NGPUS_PER_NODE MUST match.
#
# Override at submission time, e.g.:
#   sbatch --export=DISTANCES="3 5",NUM_STEPS=1000000 run_trans7_pretrain_multi_node.slurm
###############################################################################

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
set -euo pipefail

# ---- Configurable parameters (override via --export or edit here) -----------
NGPUS_PER_NODE="${NGPUS_PER_NODE:-4}"          # must match --gres=gpu: above
DISTANCES="${DISTANCES:-3}"
NUM_STEPS="${NUM_STEPS:-1000000}"
TOTAL_GPUS=$(( SLURM_NNODES * NGPUS_PER_NODE ))
BATCH_SIZE="${BATCH_SIZE:-$(( 256 / TOTAL_GPUS ))}"  # per-GPU; keeps effective batch=256
EVAL_EVERY="${EVAL_EVERY:-15000}"
D_MODEL="${D_MODEL:-256}"
ROUNDS="${ROUNDS:-1 3 5 7 9 11 13 15 17 19 21 23 25}"
USE_WANDB="${USE_WANDB:-}"
RESUME_CKPT="${RESUME_CKPT:-}"
MASTER_PORT="${MASTER_PORT:-29500}"
# ---------------------------------------------------------------------------

# ---- Distributed setup ----------------------------------------------------
# MASTER_ADDR = hostname of the first allocated node.
# SLURM sets SLURM_JOB_NODELIST; scontrol expands it to one host per line.
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR MASTER_PORT
# ---------------------------------------------------------------------------

# ---- Auto-requeue handler -------------------------------------------------
# PATH A — SIGUSR1 trap (time limit approaching):
#   Fires ~120s before wall time while srun is still running. Submits the
#   next job with the latest resume checkpoint, then this job dies at limit.
#
# PATH B — post-srun exit check:
#   If srun exits on its own before the time limit (crash or early finish),
#   we check the step in the resume checkpoint. If training is incomplete,
#   we requeue. If complete, we don't.
#
# In both cases --export=ALL carries every variable (NGPUS_PER_NODE,
# DISTANCES, NUM_STEPS, etc.) unchanged to the next job.

_REQUEUED=false

_do_requeue() {
    local CKPT_DIR="/home/${USER}/src/trans7_alphaqubit/checkpoints/pretrain"
    local BASES_STR="x+z"
    local NEW_RESUME="${CKPT_DIR}/pretrain_${BASES_STR}_d${DISTANCES}_resume.pth"
    if [ -f "$NEW_RESUME" ]; then
        echo "  Resume checkpoint: $NEW_RESUME"
        sbatch --export=ALL,RESUME_CKPT="$NEW_RESUME" "$(realpath "$0")"
        echo "  Next job submitted."
        _REQUEUED=true
    else
        echo "  WARNING: resume checkpoint not found at $NEW_RESUME"
        echo "  Resubmit manually:"
        echo "    sbatch --export=ALL,RESUME_CKPT=<path> $(realpath "$0")"
    fi
}

_signal_requeue() {
    echo "===== SIGUSR1 — time limit in ~120s at $(date) ====="
    _do_requeue
}
trap '_signal_requeue' USR1
# ---------------------------------------------------------------------------

echo "===== Job started at $(date) ====="
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "Nodes ($SLURM_NNODES): $(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')"
echo "GPUs: $SLURM_NNODES nodes × $NGPUS_PER_NODE GPUs = $TOTAL_GPUS total"
echo "MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT"
echo "DISTANCES=$DISTANCES  NUM_STEPS=$NUM_STEPS  BATCH_SIZE=$BATCH_SIZE (eff=$(( BATCH_SIZE * TOTAL_GPUS )))"

mkdir -p logs

################################################################################
# 0) Install Miniforge + conda env (identical to single-node script)
################################################################################

if [ -z "${SCRDIR:-}" ]; then
  export SCRDIR="/scratch/${USER}"
fi
mkdir -p "$SCRDIR"

MINIFORGE_DIR="$SCRDIR/miniforge3"
ENV_PREFIX="$SCRDIR/conda_envs/alphaqubit_t7_env"

if [ ! -d "$MINIFORGE_DIR" ]; then
  curl -L -o "$SCRDIR/Miniforge3.sh" \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash "$SCRDIR/Miniforge3.sh" -b -p "$MINIFORGE_DIR"
elif ! "$MINIFORGE_DIR/bin/python" -c "import boltons" &>/dev/null 2>&1; then
  echo "WARNING: boltons missing — repairing..."
  "$MINIFORGE_DIR/bin/pip" install boltons || {
    echo "ERROR: Repair failed. Another job may be using this environment."
    echo "Wait for other jobs to finish, then manually run:"
    echo "  rm -rf $MINIFORGE_DIR && resubmit"
    exit 1
  }
  echo "Repair succeeded."
fi

source "$MINIFORGE_DIR/etc/profile.d/conda.sh"

echo "Conda: $(which conda)"
conda --version

if [ ! -d "$ENV_PREFIX" ]; then
  echo "ENV_PREFIX not present — creating..."
  conda create -y -p "$ENV_PREFIX" python=3.10
fi

conda activate "$ENV_PREFIX" || { echo "conda activate failed"; exit 1; }

echo "Python: $(which python)"
python -V

################################################################################
# 1) Install dependencies (skipped if already installed)
################################################################################

cd /home/${USER}/src/trans7_alphaqubit

if [ ! -f "$ENV_PREFIX/.deps_installed" ]; then
  python -m pip install --upgrade pip wheel setuptools
  python -m pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu126
  python -m pip install -r requirements_transformer.txt
  touch "$ENV_PREFIX/.deps_installed"
fi

echo "===== Verifying torch/CUDA on master node ====="
timeout 30 python - <<'PY' || echo "WARNING: CUDA verification timed out — continuing"
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
n = torch.cuda.device_count()
print(f"GPUs visible: {n}")
for i in range(n):
    print(f"  gpu {i}: {torch.cuda.get_device_name(i)}  "
          f"{torch.cuda.get_device_properties(i).total_memory // 1024**3} GB")
PY
echo "===== Proceeding to data copy + torchrun ====="

################################################################################
# 2) Copy data to node-local /tmp on ALL nodes in parallel
#
# srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 runs the command once on
# each allocated node simultaneously, so every node gets its own local copy.
################################################################################

LOCAL_DATA_DIR="/tmp/alphaqubit_${SLURM_JOB_ID}"

echo "===== Copying pretrain.h5 to $LOCAL_DATA_DIR on all $SLURM_NNODES nodes at $(date) ====="
srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
  bash -c "mkdir -p $LOCAL_DATA_DIR && cp /home/${USER}/src/data/trans7_data/pretrain.h5 $LOCAL_DATA_DIR/pretrain.h5"
echo "===== All nodes: copy complete at $(date) ====="

################################################################################
# 3) GPU monitor on master node only
################################################################################

echo "===== Starting GPU monitor (master node only) ====="
nvidia-smi \
  --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
  --format=csv \
  -l 5 \
  >> "logs/gpu_usage_${SLURM_JOB_ID}.log" &
GPU_MONITOR_PID=$!

################################################################################
# 4) Launch multi-node DDP training via srun + torchrun
#
# srun launches one torchrun process per node. Each torchrun spawns
# NGPUS_PER_NODE worker processes. torchrun uses c10d rendezvous (no external
# server needed) to coordinate across nodes via MASTER_ADDR:MASTER_PORT.
#
# Total processes = SLURM_NNODES × NGPUS_PER_NODE = world_size seen by DDP.
################################################################################

CKPT_DIR="/home/${USER}/src/trans7_alphaqubit/checkpoints/pretrain"
mkdir -p "$CKPT_DIR"

[ -n "$RESUME_CKPT" ] && echo "  Resuming from: $RESUME_CKPT" || echo "  Starting from scratch"

RESUME_FLAG=""
[ -n "$RESUME_CKPT" ] && RESUME_FLAG="--resume $RESUME_CKPT"

echo "===== Starting multi-node DDP ($SLURM_NNODES nodes × $NGPUS_PER_NODE GPUs) at $(date) ====="

set +e
srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
  torchrun \
    --nnodes="$SLURM_NNODES" \
    --nproc_per_node="$NGPUS_PER_NODE" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --rdzv_id="$SLURM_JOB_ID" \
    run_pretrain_ddp.py \
    --distances ${DISTANCES} \
    --bases x z \
    --rounds ${ROUNDS} \
    --num_steps "${NUM_STEPS}" \
    --batch_size "${BATCH_SIZE}" \
    --eval_every "${EVAL_EVERY}" \
    --d_model "${D_MODEL}" \
    --data_dir "$LOCAL_DATA_DIR" \
    --checkpoint_dir "$CKPT_DIR" \
    ${USE_WANDB} \
    ${RESUME_FLAG}
TORCHRUN_EXIT=$?
set -e

kill $GPU_MONITOR_PID 2>/dev/null || true

echo "===== srun/torchrun exited (code=$TORCHRUN_EXIT) at $(date) ====="

################################################################################
# PATH B — post-training completion check (same logic as single-node script)
################################################################################
if [ "$_REQUEUED" = "false" ]; then
    NEW_RESUME="${CKPT_DIR}/pretrain_x+z_d${DISTANCES}_resume.pth"
    if [ -f "$NEW_RESUME" ]; then
        DONE_STEP=$(python3 - "$NEW_RESUME" <<'PY'
import sys, torch
try:
    c = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    print(c.get("step", -1))
except Exception:
    print(-1)
PY
        )
        echo "  Checkpoint step: $DONE_STEP / $NUM_STEPS"
        if [ "$DONE_STEP" -lt "$((NUM_STEPS - 1))" ]; then
            echo "  Training incomplete — submitting next job..."
            _do_requeue
        else
            echo "  Training complete at step $DONE_STEP. No requeue needed."
        fi
    else
        echo "  No resume checkpoint found. Check .err log for early crash."
        echo "  To requeue manually: sbatch --export=ALL,RESUME_CKPT=<path> $(realpath "$0")"
    fi
fi

echo "===== Job done at $(date) ====="
echo "Checkpoints: $CKPT_DIR"
