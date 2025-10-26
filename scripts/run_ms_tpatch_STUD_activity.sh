#!/usr/bin/env bash
#SBATCH --job-name=activity_ms_stud
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --array=1-5
#SBATCH --output=/home/solgi/MSP-IMTS/logs/%x_%A_%a.out
#SBATCH --error=/home/solgi/MSP-IMTS/logs/%x_%A_%a.err
#SBATCH --chdir=/home/solgi/MSP-IMTS/tPatchGNN

# === Robust execution ===
set -euo pipefail

# === Activate conda ===
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate solgiland

# === Threads setup ===
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# === Python path (for clean module imports) ===
export PYTHONPATH="/home/solgi/MSP-IMTS:/home/solgi/MSP-IMTS/tPatchGNN:${PYTHONPATH-}"

# === Variables for training ===
SEED=${SLURM_ARRAY_TASK_ID}
GPU=0
EPOCHS=600
PATIENCE=60
BATCH=32
LR=1e-3
HISTORY=3000
SCALES="100,300"
STRIDES="100,300"

echo "Running seed=$SEED on activity dataset"
echo "SCALES=$SCALES, STRIDES=$STRIDES"

python run_models.py \
  --dataset activity \
  --history $HISTORY \
  --hid_dim 32 \
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
  --fusion concat
