#!/usr/bin/env bash
#SBATCH --job-name=activity_ms_test           # Job name
#SBATCH --partition=STUD                      # TEST cluster/partition
#SBATCH --gres=gpu:1                          # Request 1 GPU
#SBATCH --cpus-per-task=2                     # Use 2 CPUs (lighter load)
#SBATCH --mem=8G                              # 8 GB memory
#SBATCH --time=6:00:00                        # 6 hours (very short runtime)
#SBATCH --output=%x_%j.out                    # STDOUT log
#SBATCH --error=%x_%j.err                     # STDERR log
#SBATCH --chdir=/home/solgi/MSP-IMTS/logs     # Where logs are written

# --- Safety ---
set -euo pipefail

# --- Environment setup ---
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate solgiland

# --- Performance settings ---
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Ensure local repo is on PYTHONPATH
export PYTHONPATH="/home/solgi/MSP-IMTS:/home/solgi/MSP-IMTS/tPatchGNN:${PYTHONPATH-}"

# --- Move to code directory ---
cd /home/solgi/MSP-IMTS/tPatchGNN

# === Configuration ===
SEED=1
GPU=0
EPOCHS=100                 # keep at 1 epoch for TEST cluster
PATIENCE=5
BATCH=4                  # smaller batch to reduce memory usage
LR=1e-3
HISTORY=3000
SCALES="300,1200"
STRIDES="150,600"

echo "============================================================"
echo "  TEST CLUSTER: 1-EPOCH sanity check for multi-scale + attention"
echo "  Dataset : activity"
echo "  Scales  : $SCALES"
echo "  Strides : $STRIDES"
echo "  Fusion  : scale_attn"
echo "============================================================"

python run_models.py \
  --dataset activity \
  --history 3000 \
  --hid_dim 64 \
  --te_dim 10 \
  --node_dim 10 \
  --nlayer 1 \
  --tf_layer 1 \
  --nhead 1 \
  --batch_size 32 \
  --lr 1e-3 \
  --w_decay 1e-4 \
  --patience 80 \
  --epoch 400 \
  --seed 1 \
  --gpu 0 \
  --multi_scales "200,1200" \
  --multi_strides "100,600" \
  --fusion scale_attn \
  --state compare_scaleattn


echo "============================================================"
echo "  TEST job finished successfully."
echo "  Check logs in: /home/solgi/MSP-IMTS/logs/activity_ms_test_<jobID>.out"
echo "============================================================"
