#!/usr/bin/env bash
#SBATCH --job-name=MH_USHCN_MS
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
EPOCHS=300            # okay to go higher; early stop handles it
PATIENCE=50           # paper
BATCH=192             # paper
LR=1e-3
HISTORY=24            # paper
# HID=32                # paper
SCALES="4,12"
STRIDES="4,12"
SEED=1

# echo "USHCN MS 2-scale (concat): scales=$SCALES strides=$STRIDES"
# for SEED in 1 2 3 4 5; do
  # echo "==== Seed $SEED ===="
python RunModelsLogsperDim.py \
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
# done
