#!/usr/bin/env bash
#SBATCH --job-name=N1_G1_D1
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --array=1
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --chdir=/home/helmy/MSP-IMTS/analyzelogs

set -euo pipefail

source /home/helmy/miniconda3/etc/profile.d/conda.sh
conda activate condaworld310
cd /home/helmy/MSP-IMTS/tPatchGNN

GPU=0
EPOCHS=300
PATIENCE=40
BATCH=64
LR=1e-3
HISTORY=3000

for SEED in 1 2 3 4 5; do
  echo "=============================="
  echo "Running with SEED=${SEED}"
  echo "=============================="

  python RunModelsSingle.py \
    --dataset activity \
    --normalization 1 \
    --denorm_test_pred 1 \
    --global_loss 1 \
    --history $HISTORY \
    --hid_dim 64 \
    --te_dim 10 \
    --node_dim 10 \
    --nlayer 1 \
    --tf_layer 1 \
    --nhead 4 \
    --batch_size $BATCH \
    --lr $LR \
    --patience $PATIENCE \
    --epoch $EPOCHS \
    --seed $SEED \
    --gpu $GPU \
    --outlayer Linear \
    --patch_size 300 \
    --stride 300
done