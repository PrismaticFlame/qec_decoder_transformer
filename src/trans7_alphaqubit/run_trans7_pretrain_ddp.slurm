#!/bin/bash
######################## SLURM OPTIONS #########################################
#SBATCH --job-name=trans7_ddp
#SBATCH --account=quantum_training
#SBATCH --time=168:00:00
#SBATCH --partition=h200_normal_q
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=logs/trans7_ddp_%j.out
#SBATCH --error=logs/trans7_ddp_%j.err
#SBATCH --signal=B:USR1@120
###############################################################################
# Trans7 AlphaQubit Pretraining — Multi-GPU (DDP, GPU options:1, 2, 4, 8)
# Note: ARC has 7 nodes of 8 H200 GPUs. Unless we feel compelled by god Herself, 
# don't use more than one node and ideally more than 4 GPUs.
#
# Uses torchrun to launch DistributedDataParallel across N H200 GPUs.
# Each GPU processes a separate data shard; gradients are synchronized
# via NCCL all-reduce.  Near-linear speedup: 2 GPUs ≈ 2x faster.
#
# Override at submission time, e.g.:
#   sbatch --export=DISTANCES="3",NUM_STEPS=500000,NGPUS=2 run_trans7_pretrain_ddp.slurm
###############################################################################

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
set -euo pipefail

# ---- Configurable parameters (override via --export or edit here) -----------
NGPUS="${NGPUS:-2}"                    # GPUs per node (must match --gres above)
DISTANCES="${DISTANCES:-3}"            # space-separated list
NUM_STEPS="${NUM_STEPS:-1000000}"
BATCH_SIZE="${BATCH_SIZE:-$(( 256 / NGPUS ))}"  # per-GPU batch; 256/NGPUS keeps effective batch=256 (matches paper)
EVAL_EVERY="${EVAL_EVERY:-15000}"
D_MODEL="${D_MODEL:-256}"
ROUNDS="${ROUNDS:-1 3 5 7 9 11 13 15 17 19 21 23 25}"
USE_WANDB="${USE_WANDB:-}"             # set to "--use_wandb" to enable
# Resume: path to a *_resume.pth checkpoint. Empty = start from scratch.
# Set automatically by the requeue handler below, or pass manually:
#   sbatch --export=ALL,RESUME_CKPT=/path/to/pretrain_x+z_d3_resume.pth run_trans7_pretrain_ddp.slurm
RESUME_CKPT="${RESUME_CKPT:-}"
# Optimizer: 'lion' (paper default) or 'adamw' (alternative).
# AdamW typically needs a higher lr than Lion; if using adamw without an explicit
# lr override, consider passing LR=3e-4 (vs Lion's 1.3e-4 for d=3).
OPTIMIZER="${OPTIMIZER:-lion}"
LR="${LR:-}"                           # empty = use TrainConfig default for the distance
# ----------------------------------------------------------------------------

# ---- Auto-requeue handler --------------------------------------------------
# There are two paths that trigger a requeue:
#
#   PATH A — Time limit (handled by signal trap):
#     SLURM sends SIGUSR1 120s before the wall time. The trap fires while
#     torchrun is still running, submits the next job, then this job gets
#     killed at the limit. _REQUEUED is set to true so PATH B is skipped.
#
#   PATH B — Clean exit before time limit (handled after torchrun returns):
#     torchrun exits on its own (training done OR crash). We read the step
#     from the resume checkpoint. If step < NUM_STEPS, training isn't done
#     and we requeue. If step >= NUM_STEPS, training is complete, no requeue.
#
# In both cases we submit with --export=ALL,RESUME_CKPT=... so every variable
# (NGPUS, DISTANCES, NUM_STEPS, etc.) carries over unchanged to the next job.
# NUM_STEPS is always the *total* target — never changes between jobs. The
# resume checkpoint tracks the current step, so each job naturally picks up
# where the last one left off. No manual "portioning" required.

_REQUEUED=false   # flag to prevent double-requeue if both paths trigger

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
    # PATH A: time limit approaching
    echo "===== SIGUSR1 — time limit in ~120s at $(date) ====="
    _do_requeue
}
trap '_signal_requeue' USR1
# ---------------------------------------------------------------------------

echo "===== Job started on $(hostname) at $(date) ====="
echo "SLURM_JOB_ID=$SLURM_JOB_ID  NGPUS=$NGPUS"
echo "Working dir: $(pwd)"
echo "DISTANCES=$DISTANCES  NUM_STEPS=$NUM_STEPS  BATCH_SIZE=$BATCH_SIZE"

mkdir -p logs

################################################################################
# 0) Install Miniforge into $SCRDIR, Create & Activate env inside $SCRDIR 
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

# initializing conda for non-interactive shell
source "$MINIFORGE_DIR/etc/profile.d/conda.sh"

echo "Conda: $(which conda)"
conda --version

echo "=== Creating conda env at $ENV_PREFIX ===="
if [ ! -d "$ENV_PREFIX" ]; then
  echo "ENV_PREFIX not present"
  conda create -y -p "$ENV_PREFIX" python=3.10
fi

conda activate "$ENV_PREFIX" || { echo "conda activate failed"; exit 1; }

echo "Python: $(which python)"
python -V

################################################################################
# 1) Install dependencies (skipped if already installed)
################################################################################

# get to working folder
cd /home/${USER}/src/trans7_alphaqubit

if [ ! -f "$ENV_PREFIX/.deps_installed" ]; then
  python -m pip install --upgrade pip wheel setuptools
  python -m pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu126
  python -m pip install -r requirements_transformer.txt
  touch "$ENV_PREFIX/.deps_installed"
fi

echo "===== Verifying torch/CUDA installation ====="
timeout 30 python - <<'PY' || echo "WARNING: CUDA verification timed out or failed — continuing anyway"
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("Initializing CUDA (may hang here if driver issues)...")
print("cuda available:", torch.cuda.is_available())
n = torch.cuda.device_count()
print(f"GPUs visible: {n}")
for i in range(n):
    print(f"  gpu {i}: {torch.cuda.get_device_name(i)}  "
          f"{torch.cuda.get_device_properties(i).total_memory // 1024**3} GB")
print("CUDA verification complete.")
PY
echo "===== Proceeding to torchrun ====="

################################################################################
# 2) Copy data to node-local storage
################################################################################

LOCAL_DATA_DIR="/tmp/alphaqubit_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_DATA_DIR"
echo "===== Copying pretrain.h5 to $LOCAL_DATA_DIR at $(date) ====="
cp "/home/${USER}/src/data/trans7_data/pretrain.h5" "$LOCAL_DATA_DIR/pretrain.h5"
echo "===== Copy complete at $(date) ====="

################################################################################
# 3) Run DDP pretraining
################################################################################

echo "===== Starting DDP pretraining ($NGPUS GPUs) at $(date) ====="
[ -n "$RESUME_CKPT" ] && echo "  Resuming from: $RESUME_CKPT" || echo "  Starting from scratch"

RESUME_FLAG=""
[ -n "$RESUME_CKPT" ] && RESUME_FLAG="--resume $RESUME_CKPT"

LR_FLAG=""
[ -n "$LR" ] && LR_FLAG="--lr $LR"

CKPT_DIR="/home/${USER}/src/trans7_alphaqubit/checkpoints/pretrain"

echo "===== Starting GPU monitor (logging every 5s to logs/gpu_usage_${SLURM_JOB_ID}.log) ====="
nvidia-smi \
  --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
  --format=csv \
  -l 5 \
  >> "logs/gpu_usage_${SLURM_JOB_ID}.log" &
GPU_MONITOR_PID=$!

# Disable -e for the torchrun block so a non-zero exit (killed by time limit,
# OOM, etc.) doesn't abort the script before the PATH B requeue check below.
set +e
torchrun \
  --standalone \
  --nproc_per_node="${NGPUS}" \
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
  --optimizer "${OPTIMIZER}" \
  ${LR_FLAG} \
  ${USE_WANDB} \
  ${RESUME_FLAG}
TORCHRUN_EXIT=$?
set -e
kill $GPU_MONITOR_PID 2>/dev/null || true

echo "===== torchrun exited (code=$TORCHRUN_EXIT) at $(date) ====="

################################################################################
# PATH B — post-torchrun completion check
# Only runs if PATH A (signal trap) did not already requeue.
# Reads the step saved in the resume checkpoint and requeues if training
# is not yet at NUM_STEPS. This handles:
#   - Training that finishes cleanly but before the time limit (shouldn't
#     normally trigger requeue — step will equal NUM_STEPS)
#   - Crashes or OOM exits where torchrun returned non-zero but a resume
#     checkpoint was written at the last eval
################################################################################
if [ "$_REQUEUED" = "false" ]; then
    NEW_RESUME="${CKPT_DIR}/pretrain_x+z_d${DISTANCES}_resume.pth"
    if [ -f "$NEW_RESUME" ]; then
        DONE_STEP=$(python3 - <<'PY'
import sys, torch
try:
    c = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    print(c.get("step", -1))
except Exception:
    print(-1)
PY
        "$NEW_RESUME")
        echo "  Checkpoint step: $DONE_STEP / $NUM_STEPS"
        if [ "$DONE_STEP" -lt "$((NUM_STEPS - 1))" ]; then
            echo "  Training incomplete — submitting next job..."
            _do_requeue
        else
            echo "  Training complete at step $DONE_STEP. No requeue needed."
        fi
    else
        # No resume checkpoint at all — either training never reached the first
        # eval, or it crashed very early. Don't requeue automatically.
        echo "  No resume checkpoint found. If this was a crash, check the .err log."
        echo "  To requeue manually: sbatch --export=ALL,RESUME_CKPT=<path> $(realpath "$0")"
    fi
fi

echo "===== Job done at $(date) ====="
echo "Checkpoints: $CKPT_DIR"
