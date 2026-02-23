#!/usr/bin/env bash
#SBATCH --job-name=MH_PHYSIO_MS_GLOSS
#SBATCH --partition=STUD
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
PATIENCE=30 #was 30
BATCH=32
LR=1e-4  #best 1e-4
HISTORY=24
QUANT=1.0
SCALES="2,8"
STRIDES="2,8" #was 2,8  #best 1,4
SEED=4


python RunModelsMultiGLoss.py \
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




