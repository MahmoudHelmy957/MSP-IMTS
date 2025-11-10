#!/usr/bin/env bash
#SBATCH --job-name=activity_ms_mixer           # Job name
#SBATCH --partition=STUD                        # Partition / queue
#SBATCH --gres=gpu:1                           # 1 GPU
#SBATCH --cpus-per-task=4                      # 4 CPU cores
#SBATCH --mem=16G                              # 16 GB RAM
#SBATCH --time=10:00:00                        # 10 hour limit (adjust as needed)
#SBATCH --output=%x_%j.out                     # STDOUT (%x=job name, %j=job ID)
#SBATCH --error=%x_%j.err                      # STDERR
#SBATCH --chdir=/home/solgi/MSP-IMTS/logs      # Working directory for logs

# --- Safety settings ---
set -euo pipefail

# --- Environment setup ---
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate solgiland

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH="/home/solgi/MSP-IMTS:/home/solgi/MSP-IMTS/tPatchGNN:${PYTHONPATH-}"

cd /home/solgi/MSP-IMTS/tPatchGNN

# === Experiment configuration ===
SEED=1
GPU=0
EPOCHS=600
PATIENCE=80
BATCH=32
LR=1e-3
HISTORY=4000
SCALES="200,1200"
STRIDES="100,600"

echo "============================================================"
echo "  Activity multi-scale + Mixer fusion experiment"
echo "  SEED=$SEED"
echo "  GPU=$GPU"
echo "  HISTORY=$HISTORY"
echo "  SCALES=$SCALES"
echo "  STRIDES=$STRIDES"
echo "============================================================"

# --- Launch training ---
python run_models.py \
  --dataset activity \
  --history ${HISTORY} \
  --hid_dim 32 \
  --te_dim 10 \
  --node_dim 10 \
  --nlayer 1 \
  --tf_layer 1 \
  --nhead 1 \
  --batch_size ${BATCH} \
  --lr ${LR} \
  --w_decay 1e-4 \
  --patience ${PATIENCE} \
  --epoch ${EPOCHS} \
  --seed ${SEED} \
  --gpu ${GPU} \
  --multi_scales ${SCALES} \
  --multi_strides ${STRIDES} \
  --fusion concat \
  --state activity_mixer

echo "============================================================"
echo "  Job finished successfully."
echo "  Logs are in: /home/solgi/MSP-IMTS/logs/"
echo "============================================================"
