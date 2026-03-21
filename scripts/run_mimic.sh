#!/usr/bin/env bash
#SBATCH --job-name=mimic_ss_global
#SBATCH --partition=NGPU
#SBATCH --nodelist=gpu-200
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --chdir=/home/ouass/Test/MSP-IMTS/mimicooo

set -euo pipefail
source $HOME/venv310/bin/activate
export PYTHONPATH="/home/ouass/Test/MSP-IMTS:/home/ouass/Test/MSP-IMTS/tPatchGNN:${PYTHONPATH-}"
cd /home/ouass/Test/MSP-IMTS/tPatchGNN


for seed in {1..5}; do
  echo "Running with seed $seed"

  python run_models.py \
    --dataset mimic --state def --history 24 \
    --patience 10 --epoch 1000 \
    --batch_size 32 --lr 1e-3 \
    --patch_size 8 --stride 8 \
    --nhead 1 --tf_layer 1 --nlayer 1 \
    --te_dim 10 --node_dim 10 --hid_dim 64 \
    --outlayer Linear --seed $seed --gpu 0 \
    --metric global \
    --save experiments/mimic_SS_ps8_s8_seed1
done
