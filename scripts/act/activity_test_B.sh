#!/usr/bin/env bash
#SBATCH --job-name=act_MS_B
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:10:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --chdir=/home/ouass/Test/MSP-IMTS/act

set -euo pipefail
source "$HOME/venv310/bin/activate"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH="/home/ouass/Test/MSP-IMTS:/home/ouass/Test/MSP-IMTS/tPatchGNN:${PYTHONPATH-}"

cd /home/ouass/Test/MSP-IMTS/tPatchGNN

SEED=1
GPU=0
EPOCHS=450
PATIENCE=90
BATCH=64
LR=1e-3
WDECAY=1e-4
HISTORY=6000

SCALES="300,2400"
STRIDES="150,600"

echo "Running ACTIVITY MS (concat): seed=$SEED scales=$SCALES strides=$STRIDES"
python run_models.py \
  --dataset activity \
  --history $HISTORY \
  --hid_dim 64 \
  --te_dim 10 \
  --node_dim 10 \
  --nlayer 1 \
  --tf_layer 1 \
  --nhead 1 \
  --batch_size $BATCH \
  --lr $LR \
  --w_decay $WDECAY \
  --patience $PATIENCE \
  --epoch $EPOCHS \
  --seed $SEED \
  --gpu $GPU \
  --multi_scales "$SCALES" \
  --multi_strides "$STRIDES" \
  --fusion concat
