#!/usr/bin/env bash
#SBATCH --job-name=physio_STUD_ms2_scaleattn
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=23:59:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --chdir=/home/ouass/Test/MSP-IMTS/logs

set -euo pipefail
source "$HOME/venv310/bin/activate"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH="/home/ouass/Test/MSP-IMTS:/home/ouass/Test/MSP-IMTS/tPatchGNN:${PYTHONPATH-}"

cd /home/ouass/Test/MSP-IMTS/tPatchGNN

SEED=1
GPU=0
EPOCHS=600
PATIENCE=60
BATCH=32
LR=1e-3
HISTORY=24
QUANT=1.0

SCALES="2,8"
STRIDES="2,8"

# scale-attn “best-bet”
AT_HIDDEN=32
AT_TEMP=1.25
AT_DROPOUT=0.10
AT_REG=5e-4
AT_NORM="--attn_norm"

echo "STUD MS2 scale-attn run:"
echo "seed=${SEED} scales=${SCALES} strides=${STRIDES} temp=${AT_TEMP} reg=${AT_REG} dropout=${AT_DROPOUT} norm=on"

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
  --attn_hidden $AT_HIDDEN \
  --attn_temp $AT_TEMP \
  --attn_dropout $AT_DROPOUT \
  --attn_reg $AT_REG
