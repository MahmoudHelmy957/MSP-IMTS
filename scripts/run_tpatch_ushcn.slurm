#!/bin/bash
#SBATCH --job-name=ushcn_seeds
#SBATCH --output=logs/ushcn_seed_%j.out
#SBATCH --error=logs/ushcn_seed_%j.err
#SBATCH --partition=STUD      # or TEST for quick debug
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=16G

export PYTHONPATH=$(pwd):$PYTHONPATH

# Run over 5 seeds
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