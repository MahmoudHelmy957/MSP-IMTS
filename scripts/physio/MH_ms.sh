#!/usr/bin/env bash
#SBATCH --job-name=MD_14
#SBATCH --partition=NGPU
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --chdir=/home/helmy/MSP-IMTS/analyzelogs

set -euo pipefail

# init conda in non-interactive shell (slurm)
source /home/helmy/miniconda3/etc/profile.d/conda.sh
# conda activation hooks sometimes reference unset vars (breaks with -u)
conda activate condaworld310
cd /home/helmy/MSP-IMTS/tPatchGNN


GPU=0
EPOCHS=300
PATIENCE=30 #was 30
BATCH=32
LR=1e-4  #best 1e-4
HISTORY=24
QUANT=1.0
SCALES="1,4"
STRIDES="1,4" #was 2,8  #best 1,4



for SEED in 1 2 3 4 5; do
  echo "=============================="
  echo "Running with SEED=${SEED}"
  echo "=============================="

  python RunModelsMultiDLoss.py \
    --dataset physionet \
    --history $HISTORY \
    --quantization $QUANT \
    --hid_dim 64 \
    --te_dim 32 \
    --node_dim 16 \
    --nlayer 2 \
    --tf_layer 2 \
    --nhead 4 \
    --batch_size $BATCH \
    --lr $LR \
    --patience $PATIENCE \
    --epoch $EPOCHS \
    --seed $SEED \
    --gpu $GPU \
    --multi_scales "$SCALES" \
    --multi_strides "$STRIDES" \
    --fusion concat




