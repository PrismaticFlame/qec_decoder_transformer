#!/bin/bash
######################## SLURM OPTIONS #########################################
#SBATCH --job-name=trans7_workers
#SBATCH --account=quantum_training
#SBATCH --time=02:00:00
#SBATCH --partition=h200_normal_q
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=logs/trans7_workers_%j.out
#SBATCH --error=logs/trans7_workers_%j.err
###############################################################################
# Trans7 AlphaQubit — Worker / Bottleneck Diagnostic Run
#
# Purpose: identify GPU, CPU, and I/O bottlenecks. Does NOT save checkpoints.
# Runs a short fixed number of steps with dense resource monitoring:
#   - GPU utilization + memory + power (nvidia-smi, ~1s intervals)
#   - GPU SM/memory bandwidth metrics (nvidia-smi dmon, ~1s intervals)
#   - Per-CPU utilization (mpstat, 5s intervals)
#   - Disk I/O throughput (iostat, 5s intervals)
#
# To test different num_workers values, edit dataset.py make_loader() default
# before submitting, since there is no CLI flag for it yet.
#
# Override at submission time, e.g.:
#   sbatch --export=DISTANCES="3",NUM_STEPS=500,NGPUS=2 run_trans7_pretrain_workers.slurm
###############################################################################

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
set -euo pipefail

# ---- Configurable parameters ------------------------------------------------
NGPUS="${NGPUS:-2}"
DISTANCES="${DISTANCES:-3}"
NUM_STEPS="${NUM_STEPS:-500}"          # short — just enough to see steady-state behavior
BATCH_SIZE="${BATCH_SIZE:-$(( 256 / NGPUS ))}"  # per-GPU batch; 256/NGPUS keeps effective batch=256 (matches paper)
EVAL_EVERY="${EVAL_EVERY:-9999999}"    # effectively disabled — we want raw training throughput
D_MODEL="${D_MODEL:-256}"
ROUNDS="${ROUNDS:-1 3 5}"             # reduced to keep it quick
USE_WANDB="${USE_WANDB:-}"
# -----------------------------------------------------------------------------

echo "===== Job started on $(hostname) at $(date) ====="
echo "SLURM_JOB_ID=$SLURM_JOB_ID  NGPUS=$NGPUS"
echo "Working dir: $(pwd)"
echo "DISTANCES=$DISTANCES  NUM_STEPS=$NUM_STEPS  BATCH_SIZE=$BATCH_SIZE"
echo "EVAL_EVERY=$EVAL_EVERY  (disabled for throughput measurement)"

mkdir -p logs

################################################################################
# 0) Conda env setup (identical health-check logic to test/ddp slurm files)
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

# Checkpoint dir in /tmp — discarded when job ends, no persistent checkpoints saved
CHECKPOINT_DIR="$LOCAL_DATA_DIR/checkpoints_discard"
mkdir -p "$CHECKPOINT_DIR"
echo "NOTE: Checkpoints written to $CHECKPOINT_DIR (in /tmp — discarded after job)"

################################################################################
# 3) Start resource monitors
################################################################################

LOG_PREFIX="logs/workers_diag_${SLURM_JOB_ID}"

echo "===== Starting resource monitors at $(date) ====="

# --- GPU utilization + memory + power (~1s intervals, approximates per-step) ---
# At full pace (~0.6s/step) this gives roughly one sample per training step.
nvidia-smi \
  --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
  --format=csv \
  -l 1 \
  >> "${LOG_PREFIX}_gpu_util.csv" &
GPU_UTIL_PID=$!
echo "GPU util monitor PID=$GPU_UTIL_PID -> ${LOG_PREFIX}_gpu_util.csv"

# --- GPU SM / memory-bandwidth / NVLINK metrics (nvidia-smi dmon) ---
# dmon gives deeper metrics: SM%, mem%, encoder/decoder%, PCIe throughput
nvidia-smi dmon \
  -s pucvmet \
  -d 1 \
  >> "${LOG_PREFIX}_gpu_dmon.log" &
GPU_DMON_PID=$!
echo "GPU dmon monitor PID=$GPU_DMON_PID -> ${LOG_PREFIX}_gpu_dmon.log"

# --- Per-CPU utilization (mpstat, 5s intervals) ---
# Shows whether DataLoader workers are saturating CPU cores.
# %iowait column reveals if workers are blocked on disk reads.
if command -v mpstat &>/dev/null; then
  mpstat -P ALL 5 >> "${LOG_PREFIX}_cpu.log" &
  CPU_PID=$!
  echo "CPU monitor PID=$CPU_PID -> ${LOG_PREFIX}_cpu.log"
else
  echo "WARNING: mpstat not found — skipping CPU monitor (install sysstat)"
  CPU_PID=""
fi

# --- Disk I/O throughput (iostat, 5s intervals) ---
# Shows MB/s read/write on each device. If /tmp is on nvme, look for that device.
# High %util on the data device = I/O bottleneck.
if command -v iostat &>/dev/null; then
  iostat -x 5 >> "${LOG_PREFIX}_io.log" &
  IO_PID=$!
  echo "I/O monitor PID=$IO_PID -> ${LOG_PREFIX}_io.log"
else
  echo "WARNING: iostat not found — skipping I/O monitor (install sysstat)"
  IO_PID=""
fi

################################################################################
# 4) Run diagnostic training
################################################################################

echo "===== Starting worker diagnostic ($NGPUS GPU) at $(date) ====="
echo "      Steps=$NUM_STEPS  BatchSize=$BATCH_SIZE  Eval disabled"
echo "      Watch ${LOG_PREFIX}_gpu_util.csv for per-step GPU usage"

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
  --checkpoint_dir "$CHECKPOINT_DIR" \
  ${USE_WANDB}

################################################################################
# 5) Stop monitors and print summary
################################################################################

kill $GPU_UTIL_PID 2>/dev/null || true
kill $GPU_DMON_PID 2>/dev/null || true
[ -n "$CPU_PID" ]  && kill $CPU_PID  2>/dev/null || true
[ -n "$IO_PID" ]   && kill $IO_PID   2>/dev/null || true

echo "===== Diagnostic run finished at $(date) ====="
echo ""
echo "===== GPU UTILIZATION SUMMARY ====="
# Average GPU util % across all samples (col 3 = utilization.gpu)
awk -F',' 'NR>1 && $3 ~ /[0-9]/ { sum+=$3; count++ } END { if(count>0) printf "  Avg GPU util: %.1f%%\n", sum/count }' \
  "${LOG_PREFIX}_gpu_util.csv" || true

echo ""
echo "===== DIAGNOSTIC FILES ====="
echo "  GPU utilization : ${LOG_PREFIX}_gpu_util.csv"
echo "  GPU dmon metrics: ${LOG_PREFIX}_gpu_dmon.log"
echo "  CPU per-core    : ${LOG_PREFIX}_cpu.log"
echo "  Disk I/O        : ${LOG_PREFIX}_io.log"
echo ""
echo "To interpret results:"
echo "  GPU util <70% consistently  -> data loading or CPU is the bottleneck"
echo "  %iowait high in cpu.log     -> disk I/O bottleneck (even from /tmp)"
echo "  CPU cores near 100%         -> num_workers too high or preprocessing heavy"
echo "  GPU util near 100%          -> GPU is the bottleneck (good — increase batch)"
