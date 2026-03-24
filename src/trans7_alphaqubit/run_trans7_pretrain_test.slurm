#!/bin/bash
######################## SLURM OPTIONS #########################################
#SBATCH --job-name=trans7_test
#SBATCH --account=quantum_training
#SBATCH --time=00:30:00
#SBATCH --partition=a100_normal_q
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=logs/trans7_test_%j.out
#SBATCH --error=logs/trans7_test_%j.err
###############################################################################
# Trans7 AlphaQubit Pretraining — Smoke Test (1 GPU)
#
# Reduced job for validating script structure and env setup before
# submitting the full training run. Not intended for actual training.
#
# Override at submission time, e.g.:
#   sbatch --export=DISTANCES="3",NUM_STEPS=100,NGPUS=1 run_trans7_pretrain_test.slurm
###############################################################################

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
set -euo pipefail

# ---- Configurable parameters (override via --export or edit here) -----------
NGPUS="${NGPUS:-2}"                    # GPUs per node (must match --gres above)
DISTANCES="${DISTANCES:-3}"            # space-separated list
NUM_STEPS="${NUM_STEPS:-100}"          # reduced for smoke test
BATCH_SIZE="${BATCH_SIZE:-$(( 256 / NGPUS ))}"  # per-GPU batch; 256/NGPUS keeps effective batch=256 (matches paper)
EVAL_EVERY="${EVAL_EVERY:-50}"         # reduced for smoke test
D_MODEL="${D_MODEL:-256}"
ROUNDS="${ROUNDS:-1 3 5}"             # reduced for smoke test
USE_WANDB="${USE_WANDB:-}"             # set to "--use_wandb" to enable
# ----------------------------------------------------------------------------

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

cd /home/${USER}/src/trans7_alphaqubit

if [ ! -f "$ENV_PREFIX/.deps_installed" ]; then
  python -m pip install --upgrade pip wheel setuptools
  python -m pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu126
  python -m pip install -r requirements_transformer.txt
  touch "$ENV_PREFIX/.deps_installed"
fi

python - <<'PY'
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

################################################################################
# 2) Copy data to node-local storage
################################################################################

LOCAL_DATA_DIR="/tmp/alphaqubit_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_DATA_DIR"
echo "===== Copying pretrain.h5 to $LOCAL_DATA_DIR at $(date) ====="
cp "/home/${USER}/src/data/trans7_data/pretrain.h5" "$LOCAL_DATA_DIR/pretrain.h5"
echo "===== Copy complete at $(date) ====="

################################################################################
# 3) Run smoke test
################################################################################

echo "===== Starting GPU monitor (logging every 5s to logs/gpu_usage_${SLURM_JOB_ID}.log) ====="
nvidia-smi \
  --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
  --format=csv \
  -l 5 \
  >> "logs/gpu_usage_${SLURM_JOB_ID}.log" &
GPU_MONITOR_PID=$!

echo "===== Starting smoke test ($NGPUS GPU) at $(date) ====="

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
  --checkpoint_dir "/home/${USER}/src/trans7_alphaqubit/checkpoints/test" \
  ${USE_WANDB}

kill $GPU_MONITOR_PID 2>/dev/null || true

echo "===== Smoke test finished at $(date) ====="
echo "Checkpoints written to: /home/${USER}/src/trans7_alphaqubit/checkpoints/test"
