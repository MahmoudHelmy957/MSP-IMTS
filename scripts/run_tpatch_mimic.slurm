#!/bin/bash
#SBATCH --job-name=mimic_tpatchgnn_job
#SBATCH --output=logs/mimic_%j.out
#SBATCH --error=logs/mimic_%j.err
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --mem=16G

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tpatchgnn

# Move to project directory so Python can find local modules
cd ~/tPatchGNN/tPatchGNN

# Run over 5 seeds
for seed in {1..5}; do
  echo "Running with seed $seed"

  python run_models.py \
    --dataset mimic \
    --state def \
    --history 24 \
    --patience 10 \
    --batch_size 8 \
    --lr 1e-3 \
    --patch_size 8 \
    --stride 8 \
    --nhead 1 \
    --tf_layer 1 \
    --nlayer 1 \
    --te_dim 10 \
    --node_dim 10 \
    --hid_dim 64 \
    --outlayer Linear \
    --seed "$seed" \
    --gpu 0

done
