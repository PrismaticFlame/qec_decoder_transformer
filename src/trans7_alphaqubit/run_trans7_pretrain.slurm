#!/bin/bash
######################## SLURM OPTIONS #########################################
#SBATCH --job-name=trans7_pretrain
#SBATCH --account=quantum_training
#SBATCH --time=48:00:00
#SBATCH --partition=a100_normal_q
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/trans7_pretrain_%j.out
#SBATCH --error=logs/trans7_pretrain_%j.err
###############################################################################
# Trans7 AlphaQubit Pretraining
#
# Trains one model per distance (d=3, d=5) on all bases (X+Z) combined.
# Default: 2M steps, d_model=256, batch_init=256 -> batch_final=1024.
#
# Approximate wall time:
#   d=3 only:   ~20-30 h for 2M steps on A100
#   d=3 + d=5:  ~40-50 h (sequential)  -- increase --time if needed
#
# Override at submission time, e.g.:
#   sbatch --export=DISTANCES="3",NUM_STEPS=500000 run_trans7_pretrain.slurm
###############################################################################

export PYTHONUNBUFFERED=1
set -euo pipefail

# ---- Configurable parameters (override via --export or edit here) -----------
DISTANCES="${DISTANCES:-3 5}"          # space-separated list
NUM_STEPS="${NUM_STEPS:-2000000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EVAL_EVERY="${EVAL_EVERY:-5000}"
D_MODEL="${D_MODEL:-256}"
ROUNDS="${ROUNDS:-1 3 5 7 9 11 13 15 17 19 21 23 25}"
USE_WANDB="${USE_WANDB:-}"             # set to "--use_wandb" to enable
# ----------------------------------------------------------------------------

echo "===== Job started on $(hostname) at $(date) ====="
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
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
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("gpu memory:", torch.cuda.get_device_properties(0).total_memory // 1024**3, "GB")
PY

################################################################################
# 2) Run pretraining
################################################################################
SCRIPT_DIR="${REPO_ROOT}/src/trans7_alphaqubit"

echo "===== Starting pretraining at $(date) ====="

# run_pretrain.py uses relative imports, so we cd into its directory
cd "${SCRIPT_DIR}"

srun -n 1 python -u run_pretrain.py \
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

echo "===== Pretraining finished at $(date) ====="
echo "Checkpoints written to: ${SCRIPT_DIR}/checkpoints/pretrain/"
