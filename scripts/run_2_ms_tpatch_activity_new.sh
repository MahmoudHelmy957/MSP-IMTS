#!/usr/bin/env bash
#SBATCH --job-name=activity_ms_2scale
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --chdir=/home/solgi/MSP-IMTS/logs

set -euo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate solgiland

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH="/home/solgi/MSP-IMTS:/home/solgi/MSP-IMTS/tPatchGNN:${PYTHONPATH-}"

cd /home/solgi/MSP-IMTS/tPatchGNN

# === Experiment config ===
SEED=1
GPU=0
EPOCHS=600
PATIENCE=60
BATCH=32
LR=1e-3
HISTORY=3000
SCALES="300,1200"
STRIDES="150,600"

echo "Running multi-scale test on activity dataset (2 scales)"
echo "SEED=$SEED, SCALES=$SCALES, STRIDES=$STRIDES"

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
