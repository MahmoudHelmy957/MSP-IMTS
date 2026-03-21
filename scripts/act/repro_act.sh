#!/usr/bin/env bash
#SBATCH --job-name=activity_ss_1500
#SBATCH --partition=NGPU
#SBATCH --nodelist=gpu-200
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --chdir=/home/ouass/Test/MSP-IMTS/actiti

set -euo pipefail

source "$HOME/venv310/bin/activate"


export PYTHONPATH="$HOME/Test/MSP-IMTS:$HOME/Test/MSP-IMTS/tPatchGNN:${PYTHONPATH-}"

cd "$HOME/Test/MSP-IMTS/tPatchGNN"

gpu=0
patience=10

for seed in {1..5}; do
  echo "=== Activity | seed $seed ==="
  python run_models.py \
    --dataset activity --state def --history 3000 \
    --patience $patience --batch_size 32 --lr 1e-3 \
    --patch_size 300 --stride 300 \
    --nhead 1 --tf_layer 1 --nlayer 1 \
    --te_dim 10 --node_dim 10 --hid_dim 32 \
    --outlayer Linear \
    --metric per_dim \
    --seed $seed --gpu $gpu 
done
