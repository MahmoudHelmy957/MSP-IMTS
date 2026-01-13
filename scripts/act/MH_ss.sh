#!/usr/bin/env bash
#SBATCH --job-name=MH_ACTIVITY_SS_GNorm_DLoss
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
PATIENCE=40
BATCH=64
LR=1e-3
HISTORY=3000

for SEED in 1 2 3 4 5; do
  echo "=============================="
  echo "Running with SEED=${SEED}"
  echo "=============================="


    python RunModelsSingleDLoss.py \
      --dataset activity \
      --normalization 1\
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


# python RunModelsSingle.py \
#    --dataset activity --state 'def' --history 3000 \
#     --patience $PATIENCE --batch_size 32 --lr 1e-3 \
#     --patch_size 300 --stride 300 --nhead 1 --tf_layer 1 --nlayer 1 \
#     --te_dim 10 --node_dim 10 --hid_dim 32 \
#     --outlayer Linear --seed $SEED --gpu $GPU 

#nhead best was 4 