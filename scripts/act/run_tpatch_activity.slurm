#!/bin/bash
#SBATCH --job-name=activity_seeds
#SBATCH --output=logs/activity_seed_%j.out
#SBATCH --error=logs/activity_seed_%j.err
#SBATCH --partition=STUD      # or TEST for quick debug
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=16G

source ~/.bashrc
conda activate solgiland   # or source .venv/bin/activate if using virtualenv
cd ~/MSP-IMTS

# Run over 5 seeds
for seed in {1..5}; do
  echo "Running with seed $seed"

  python tPatchGNN/scripts/run_models.py \
    --dataset activity \
    --state def \
    --history 3000 \
    --patience 10 \
    --batch_size 32 \
    --lr 1e-3 \
    --patch_size 300 \
    --stride 300 \
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
