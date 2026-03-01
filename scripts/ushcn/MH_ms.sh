#!/usr/bin/env bash
#SBATCH --job-name=MD_014
#SBATCH --partition=NGPU
#SBATCH --nodelist=gpu-102
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
EPOCHS=300            # okay to go higher; early stop handles it
PATIENCE=30           # paper
BATCH=192             # paper
LR=1e-3
HISTORY=24            # paper
# HID=32                # paper
SCALES="1,4"
STRIDES="1,4"



for SEED in 1 2 3 4 5; do


  echo "=============================="
  echo "Running with SEED=${SEED}"
  echo "=============================="
  python RunModelsMultiDLoss.py \
    --dataset ushcn \
    --state def \
    --history $HISTORY \
    --batch_size $BATCH \
    --lr $LR \
    --patience $PATIENCE \
    --epoch $EPOCHS \
    --nhead 4 --tf_layer 2 --nlayer 2 \
    --te_dim 32 --node_dim 10 --hid_dim 64 \
    --seed $SEED --gpu $GPU \
    --multi_scales "$SCALES" \
    --multi_strides "$STRIDES" \
    --fusion concat

done
