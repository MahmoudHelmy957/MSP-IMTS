#!/usr/bin/env bash
#SBATCH --job-name=MIMIC_MS_same
#SBATCH --partition=NGPU
#SBATCH --gres=gpu:1
#SBATCH --array=1
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
PATIENCE=30
BATCH=32 #changed 32 to 64 
LR=1e-3  #best 1e-4
HISTORY=24
QUANT=1.0
SCALES="2,8"
STRIDES="2,8" #was 2,8  #best 1,4


for SEED in 1 2 3 4 5; do
  echo "=============================="
  echo "Running with SEED=${SEED}"
  echo "=============================="

  python RunModelsMultiGLoss.py \
    --dataset mimic \
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

done