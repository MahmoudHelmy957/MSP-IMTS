#!/usr/bin/env bash
#SBATCH --job-name=ushcn_ms_2scale
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --chdir=/home/ouass/Test/MSP-IMTS/logs/ushcn

set -euo pipefail
source "$HOME/venv310/bin/activate"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH="/home/ouass/Test/MSP-IMTS:/home/ouass/Test/MSP-IMTS/tPatchGNN:${PYTHONPATH-}"

cd /home/ouass/Test/MSP-IMTS/tPatchGNN

GPU=0
EPOCHS=400            # okay to go higher; early stop handles it
PATIENCE=20           # paper
BATCH=192             # paper
LR=1e-3
HISTORY=24            # paper
HID=32                # paper

# 2-scale: short + mid-seasonal
SCALES="1,2"
# STRIDES="1,3"
STRIDES="1,1"

echo "USHCN MS 2-scale (concat): scales=$SCALES strides=$STRIDES"
for SEED in 1 2 3 4 5; do
  echo "==== Seed $SEED ===="
  python run_models.py \
    --dataset ushcn \
    --state def \
    --history $HISTORY \
    --batch_size $BATCH \
    --lr $LR \
    --patience $PATIENCE \
    --epoch $EPOCHS \
    --nhead 1 --tf_layer 1 --nlayer 1 \
    --te_dim 10 --node_dim 10 --hid_dim $HID \
    --outlayer Linear \
    --seed $SEED --gpu $GPU \
    --multi_scales "$SCALES" \
    --multi_strides "$STRIDES" \
    --fusion concat
done
