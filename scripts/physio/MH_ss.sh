#!/usr/bin/env bash
#SBATCH --job-name=MH_PHYSIO_SS_Gloss
#SBATCH --partition=TEST
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
BATCH=64
LR=0.0001  #best 1e-4
HISTORY=24
QUANT=1.0
# SCALES="2,8"
# STRIDES="2,8" #was 2,8  #best 1,4

for SEED in 1 2 3 4 5; do
  echo "=============================="
  echo "Running with SEED=${SEED}"
  echo "=============================="

  python RunModelsSingleGLoss.py \
      --dataset physionet --state 'def' --quantization $QUANT --history 24 \
      --patience $PATIENCE --batch_size 32 --lr 1e-3 \
      --patch_size 8 --stride 8 --nhead 1 --tf_layer 1 --nlayer 1 \
      --te_dim 10 --node_dim 10 --hid_dim 64 \
      --outlayer Linear --seed $SEED --gpu $GPU

done
#nhead best was 4 