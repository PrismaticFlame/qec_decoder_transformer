#!/bin/bash
######################## SLURM OPTIONS #########################################
#SBATCH --job-name=trans7_ddp
#SBATCH --account=quantum_training
#SBATCH --time=48:00:00
#SBATCH --partition=a100_normal_q
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=logs/trans7_ddp_%j.out
#SBATCH --error=logs/trans7_ddp_%j.err
###############################################################################
# Trans7 AlphaQubit Pretraining — Multi-GPU (DDP, 2 GPUs)
#
# Uses torchrun to launch DistributedDataParallel across 2 A100 GPUs.
# Each GPU processes a separate data shard; gradients are synchronized
# via NCCL all-reduce.  Near-linear speedup: 2 GPUs ≈ 2x faster.
#
# Override at submission time, e.g.:
#   sbatch --export=DISTANCES="3",NUM_STEPS=500000,NGPUS=2 run_trans7_pretrain_ddp.slurm
###############################################################################

export PYTHONUNBUFFERED=1
set -euo pipefail

# ---- Configurable parameters (override via --export or edit here) -----------
NGPUS="${NGPUS:-2}"                    # GPUs per node (must match --gres above)
DISTANCES="${DISTANCES:-3 5}"          # space-separated list
NUM_STEPS="${NUM_STEPS:-2000000}"
BATCH_SIZE="${BATCH_SIZE:-256}"        # per-GPU batch size
EVAL_EVERY="${EVAL_EVERY:-5000}"
D_MODEL="${D_MODEL:-256}"
ROUNDS="${ROUNDS:-1 3 5 7 9 11 13 15 17 19 21 23 25}"
USE_WANDB="${USE_WANDB:-}"             # set to "--use_wandb" to enable
# ----------------------------------------------------------------------------

echo "===== Job started on $(hostname) at $(date) ====="
echo "SLURM_JOB_ID=$SLURM_JOB_ID  NGPUS=$NGPUS"
echo "DISTANCES=$DISTANCES  NUM_STEPS=$NUM_STEPS  BATCH_SIZE=$BATCH_SIZE"

mkdir -p logs

################################################################################
# 0) Module / conda setup
################################################################################
if ! command -v module &>/dev/null; then
  if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
  fi
fi

module purge || true
module --ignore_cache load Miniconda3/24.7.1-0

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

ENV_PREFIX="${SCRATCH:-$HOME}/conda_envs/trans7_env"
mkdir -p "$(dirname "$ENV_PREFIX")"

if [ ! -d "$ENV_PREFIX" ]; then
  echo "Creating conda env at $ENV_PREFIX ..."
  conda create -y -p "$ENV_PREFIX" python=3.10
fi

conda activate "$ENV_PREFIX"

echo "Python: $(which python)"
python -V

################################################################################
# 1) Install dependencies (skipped if already installed)
################################################################################
python -m pip install --upgrade pip wheel setuptools --quiet

python -m pip install torch \
  --index-url https://download.pytorch.org/whl/cu126 \
  --quiet

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
python -m pip install -r "${REPO_ROOT}/requirements_transformer.txt" --quiet

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
n = torch.cuda.device_count()
print(f"GPUs visible: {n}")
for i in range(n):
    print(f"  gpu {i}: {torch.cuda.get_device_name(i)}  "
          f"{torch.cuda.get_device_properties(i).total_mem // 1024**3} GB")
PY

################################################################################
# 2) Run DDP pretraining
################################################################################
SCRIPT_DIR="${REPO_ROOT}/src/trans7_alphaqubit"

echo "===== Starting DDP pretraining ($NGPUS GPUs) at $(date) ====="

cd "${SCRIPT_DIR}"

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
  --data_dir "../../data/trans7_data" \
  --checkpoint_dir "checkpoints/pretrain" \
  ${USE_WANDB}

echo "===== DDP pretraining finished at $(date) ====="
echo "Checkpoints written to: ${SCRIPT_DIR}/checkpoints/pretrain/"
