#!/usr/bin/env bash
#SBATCH --job-name=ushcn_BASE_allseeds
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --chdir=/home/ouass/Test/MSP-IMTS/logs

set -euo pipefail
source /home/ouass/venv310/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH="/home/ouass/Test/MSP-IMTS:/home/ouass/Test/MSP-IMTS/tPatchGNN:${PYTHONPATH-}"

cd /home/ouass/Test/MSP-IMTS/tPatchGNN

EPOCHS=1000
PATIENCE=10
BATCH=192
LR=1e-3
GPU=0

echo "=== USHCN baseline: ps=2, st=2, hid_dim=32, batch=192 ==="

for seed in {1..5}; do
  echo "Running with seed $seed"

  python tPatchGNN/run_models.py \
    --dataset ushcn \
    --state def \
    --history 24 \
    --patience 10 \
    --batch_size 192 \
    --lr 1e-3 \
    --patch_size 2 \
    --stride 2 \
    --nhead 1 \
    --tf_layer 1 \
    --nlayer 1 \
    --te_dim 10 \
    --node_dim 10 \
    --hid_dim 32 \
    --outlayer Linear \
    --seed "$seed" \
    --gpu 0

done