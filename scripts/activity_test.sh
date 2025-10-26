#!/usr/bin/env bash
#SBATCH --job-name=activity_ms2_concat
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --chdir=/home/ouass/Test/MSP-IMTS/logs

set -euo pipefail

# ==== env ====
source "$HOME/venv310/bin/activate"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH="/home/ouass/Test/MSP-IMTS:/home/ouass/Test/MSP-IMTS/tPatchGNN:${PYTHONPATH-}"


cd /home/ouass/Test/MSP-IMTS/tPatchGNN

SEED=1
GPU=0
EPOCHS=300
PATIENCE=40
BATCH=64         # use 32 if OOM
LR=1e-3
HISTORY=3000     # ms

SCALES="60,300"  # ms
STRIDES="30,150" # ms (50% overlap)

echo "Activity MS2 concat | seed=$SEED scales=$SCALES strides=$STRIDES"

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
