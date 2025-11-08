#!/usr/bin/env bash
#SBATCH --job-name=physio_ms_scaleattn       # Job name
#SBATCH --partition=TEST                     # Partition (use TEST first for a quick check)
#SBATCH --gres=gpu:1                         # Request one GPU
#SBATCH --cpus-per-task=4                    # 4 CPU cores
#SBATCH --mem=16G                            # 16 GB memory
#SBATCH --time=01:00:00                      # 1 hour wall time
#SBATCH --output=%x_%j.out                   # STDOUT log (%x = job name, %j = job ID)
#SBATCH --error=%x_%j.err                    # STDERR log
#SBATCH --chdir=/home/solgi/MSP-IMTS/logs    # Directory for log files

# --- Safety ---
set -euo pipefail

# --- Environment setup ---
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate solgiland

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH="/home/solgi/MSP-IMTS:/home/solgi/MSP-IMTS/tPatchGNN:${PYTHONPATH-}"

# --- Move to the training code directory ---
cd /home/solgi/MSP-IMTS/tPatchGNN

# === Experiment configuration ===
SEED=1
GPU=0
EPOCHS=600
PATIENCE=60
BATCH=16
LR=1e-3
HISTORY=24
QUANT=0.0

# Multi-scale parameters for PhysioNet (same as your best concat run)
SCALES="2,8"
STRIDES="2,8"

echo "============================================================"
echo "  PhysioNet multi-scale + attention fusion experiment"
echo "  SEED=$SEED"
echo "  SCALES=$SCALES  STRIDES=$STRIDES"
echo "============================================================"

# --- Launch training ---
python run_models.py \
  --dataset physionet \
  --history $HISTORY \
  --quantization $QUANT \
  --hid_dim 64 \
  --te_dim 10 \
  --node_dim 10 \
  --nlayer 1 \
  --tf_layer 1 \
  --nhead 1 \
  --batch_size $BATCH \
  --lr $LR \
  --patience $PATIENCE \
  --epoch $EPOCHS \
  --seed $SEED \
  --gpu $GPU \
  --multi_scales "$SCALES" \
  --multi_strides "$STRIDES" \
  --fusion scale_attn \
  --state physio_scaleattn

echo "============================================================"
echo "  Job finished. Check logs in /home/solgi/MSP-IMTS/logs/"
echo "============================================================"
