#!/usr/bin/env bash
#SBATCH --job-name=mimic_MS_4-8
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --chdir=/home/ouass/Test/MSP-IMTS/logs

set -euo pipefail
source $HOME/venv310/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH="/home/ouass/Test/MSP-IMTS:/home/ouass/Test/MSP-IMTS/tPatchGNN:${PYTHONPATH-}"
cd /home/ouass/Test/MSP-IMTS/tPatchGNN

for seed in {1..5}; do
  echo "Running with seed $seed"

  python run_models.py \
    --dataset mimic --state def --history 24 \
    --patience 20 --epoch 1000 \
    --batch_size 4 --lr 1e-3 \
    --nhead 1 --tf_layer 1 --nlayer 1 \
    --te_dim 10 --node_dim 10 --hid_dim 64 \
    --outlayer Linear --seed "$seed" --gpu 0 \
    --multi_scales 4,8 --multi_strides 4,8 --fusion concat \
    --save experiments/mimic_MS_2-8_seed1
done