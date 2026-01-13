#!/usr/bin/env bash
#SBATCH --job-name=MH_MIMIC_SS
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

# Seed from array task id (fallback to 1 if not running as an array)
SEED=${SLURM_ARRAY_TASK_ID:-1}

GPU=0
EPOCHS=500
PATIENCE=10 #was 30
BATCH=4
LR=0.0001  #best 1e-4
HISTORY=24
QUANT=1.0
# SCALES="2,8"
# STRIDES="2,8" #was 2,8  #best 1,4


python RunModelsSingle.py \
  --dataset mimic \
  --history 24 \
  --quantization $QUANT \
  --patience $PATIENCE \
  --batch_size $BATCH\
  --lr $LR \
  --patch_size 8\
  --stride 8\
  --nhead 1\
   --tf_layer 1 --nlayer 1 \
  --te_dim 10 --node_dim 10 --hid_dim 64 \
  --outlayer Linear --seed $SEED --gpu $GPU
#nhead best was 4 