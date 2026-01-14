#!/usr/bin/env bash
#SBATCH --job-name=MH_USHCN_SS_DLOSS
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
EPOCHS=500
PATIENCE=10 #was 30
BATCH=32
LR=0.0001  #best 1e-4
HISTORY=24
QUANT=1.0


for SEED in 1 2 3 4 5; do


  echo "=============================="
  echo "Running with SEED=${SEED}"
  echo "=============================="

  python RunModelsSingleDLoss.py \
      --dataset ushcn --state 'def' --history 24 \
      --patience $PATIENCE --batch_size 256 --lr 1e-3 \
      --patch_size 2 --stride 2 --nhead 4 --tf_layer 1 --nlayer 1 \
      --te_dim 32 --node_dim 10 --hid_dim 64 \
      --outlayer Linear --seed $SEED --gpu $GPU


done

