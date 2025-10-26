#!/usr/bin/env bash
#SBATCH --job-name=physio_STUD_ms_2_8_scaleattn
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=1-5
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --chdir=/home/ouass/Test/MSP-IMTS/logs

set -euo pipefail

# ==== Environment ====
source "$HOME/venv310/bin/activate"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH="/home/ouass/Test/MSP-IMTS:/home/ouass/Test/MSP-IMTS/tPatchGNN:${PYTHONPATH-}"

cd /home/ouass/Test/MSP-IMTS/tPatchGNN

# ==== Run config ====
SEED=${SLURM_ARRAY_TASK_ID}
GPU=0
EPOCHS=600
PATIENCE=60
BATCH=32
LR=1e-3
HISTORY=24
QUANT=1.0

# Multi-scale windows (hours) and strides (hours)
SCALES="2,8"
STRIDES="2,8"

# Fusion & attention hyperparams
FUSION="scale_attn"         # change to "concat" to try simple concatenation
ATTN_HIDDEN=32
ATTN_TEMP=1.0
ATTN_DROPOUT=0.0
ATTN_NORM="--attn_norm"     # set to "" to disable LayerNorm on fused features
ATTN_REG=1e-3               # entropy regularization weight on scale weights

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
python -V
echo "STUD MS run: seed=$SEED scales=$SCALES strides=$STRIDES fusion=$FUSION"
echo "Attn params: hidden=$ATTN_HIDDEN temp=$ATTN_TEMP dropout=$ATTN_DROPOUT norm=${ATTN_NORM:-off} reg=$ATTN_REG"

# ==== Launch ====
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
  --fusion $FUSION \
  --attn_hidden $ATTN_HIDDEN \
  --attn_temp $ATTN_TEMP \
  --attn_dropout $ATTN_DROPOUT \
  ${ATTN_NORM:+--attn_norm} \
  --attn_reg $ATTN_REG
