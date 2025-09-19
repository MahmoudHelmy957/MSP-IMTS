#!/usr/bin/env bash
#SBATCH --job-name=physio_STUD_ms_2_8
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=1-5
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --chdir=/home/ouass/Test/MSP-IMTS/logs

set -euo pipefail
source "$HOME/venv310/bin/activate"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH="/home/ouass/Test/MSP-IMTS:/home/ouass/Test/MSP-IMTS/tPatchGNN:${PYTHONPATH-}"

cd /home/ouass/Test/MSP-IMTS/tPatchGNN

SEED=${SLURM_ARRAY_TASK_ID}
GPU=0
EPOCHS=600
PATIENCE=60
BATCH=32
LR=1e-3
HISTORY=24
QUANT=1.0

SCALES="2,8"
STRIDES="2,8"

echo "STUD MS run: seed=$SEED scales=$SCALES strides=$STRIDES"

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
  --fusion concat
