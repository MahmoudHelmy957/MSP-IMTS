#!/usr/bin/env bash
#SBATCH --job-name=act_repro_ss_24ms
#SBATCH --partition=TEST
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --chdir=/home/ouass/Test/MSP-IMTS/act

set -euo pipefail

# ==== env ====
source "$HOME/venv310/bin/activate"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH="/home/ouass/Test/MSP-IMTS:/home/ouass/Test/MSP-IMTS/tPatchGNN:${PYTHONPATH-}"


cd /home/ouass/Test/MSP-IMTS/tPatchGNN

SEED=1
GPU=0
EPOCHS=300
PATIENCE=40
BATCH=64
LR=1e-3

# ms units; history=3000 ms, pred_window=1000 ms 
HISTORY=3000
PATCH_SIZE=24       # 24 ms windows
STRIDE=24           # 24 ms stride

echo "Reproducing Activity single-scale baseline:"
echo "seed=$SEED history=${HISTORY}ms patch=${PATCH_SIZE}ms stride=${STRIDE}ms B=$BATCH lr=$LR"

python run_models.py \
  --dataset activity \
  --history $HISTORY \
  --patch_size $PATCH_SIZE \
  --stride $STRIDE \
  --hid_dim 32 \
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
  --fusion concat 