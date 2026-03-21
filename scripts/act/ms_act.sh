#!/usr/bin/env bash
#SBATCH --job-name=activity_ms_ovlp_250_500_1000
#SBATCH --partition=NGPU
#SBATCH --nodelist=gpu-200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30000
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --chdir=/home/ouass/Test/MSP-IMTS/actiti/attn

set -euo pipefail

source "$HOME/venv310/bin/activate"


export PYTHONPATH="$HOME/Test/MSP-IMTS:$HOME/Test/MSP-IMTS/tPatchGNN:${PYTHONPATH-}"

cd "$HOME/Test/MSP-IMTS/tPatchGNN"

gpu=0
patience=20

for seed in {1..5}; do
  echo "=== Activity | seed $seed ==="
  python run_models.py \
  --dataset activity --state def --history 3000 \
  --batch_size 32 --lr 1e-3 \
  --nhead 1 --patience 10 --tf_layer 1 --nlayer 1 \
  --te_dim 10 --node_dim 10 --hid_dim 34 \
  --outlayer Linear --seed $seed --gpu 0 \
  --multi_scales 250,500,1000 --multi_strides 125,250,500 \
  --metric per_dim \
  --fusion attn
done
