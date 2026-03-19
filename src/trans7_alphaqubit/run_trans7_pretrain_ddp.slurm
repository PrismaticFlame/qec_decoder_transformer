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
###############################################################################
# Trans7 AlphaQubit Pretraining — Multi-GPU (DDP, 2 GPUs)
#
# Uses torchrun to launch DistributedDataParallel across 2 A100 GPUs.
# Each GPU processes a separate data shard; gradients are synchronized
# via NCCL all-reduce.  Near-linear speedup: 2 GPUs ≈ 4x faster.
#
# Override at submission time, e.g.:
#   sbatch --export=DISTANCES="3",NUM_STEPS=500000,NGPUS=2 run_trans7_pretrain_ddp.slurm
###############################################################################

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
set -euo pipefail

# ---- Configurable parameters (override via --export or edit here) -----------
NGPUS="${NGPUS:-2}"                    # GPUs per node (must match --gres above)
DISTANCES="${DISTANCES:-3}"          # space-separated list
NUM_STEPS="${NUM_STEPS:-1000000}"
BATCH_SIZE="${BATCH_SIZE:-256}"        # per-GPU batch size
EVAL_EVERY="${EVAL_EVERY:-15000}"
D_MODEL="${D_MODEL:-256}"
ROUNDS="${ROUNDS:-1 3 5 7 9 11 13 15 17 19 21 23 25}"
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
cd /home/cdw24/src/trans7_alphaqubit 

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
  --checkpoint_dir "/home/${USER}/src/trans7_alphaqubit/checkpoints/pretrain" \
  ${USE_WANDB}

echo "===== DDP pretraining finished at $(date) ====="
echo "Checkpoints written to: /home/${USER}/src/trans7_alphaqubit/checkpoints/pretrain"
