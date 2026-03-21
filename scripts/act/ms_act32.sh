#!/usr/bin/env bash
#SBATCH --job-name=activity_ms_750_1500
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --chdir=/home/ouass/Test/MSP-IMTS/pres/newact

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
    --patience 10 --batch_size 32 --lr 1e-3 \
    --nhead 1 --tf_layer 1 --nlayer 1 \
    --te_dim 10 --node_dim 10 --hid_dim 32 \
    --outlayer Linear \
    --multi_scales 750,1500 \
    --multi_strides 750,1500 \
    --seed $seed --gpu 0 \
    --fusion concat
done
