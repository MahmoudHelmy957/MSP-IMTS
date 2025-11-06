#!/usr/bin/env bash
#SBATCH --job-name=mimic_MS_2-8
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --chdir=/home/ouass/Test/MSP-IMTS/logs

set -euo pipefail
source $HOME/venv310/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH="/home/ouass/Test/MSP-IMTS:/home/ouass/Test/MSP-IMTS/tPatchGNN:${PYTHONPATH-}"
cd /home/ouass/Test/MSP-IMTS/tPatchGNN

python run_models.py \
  --dataset mimic --state def --history 24 \
  --patience 20 --epoch 1000 \
  --batch_size 4 --lr 1e-3 \
  --nhead 1 --tf_layer 1 --nlayer 1 \
  --te_dim 10 --node_dim 10 --hid_dim 64 \
  --outlayer Linear --seed 1 --gpu 0 \
  --multi_scales 2,8 --multi_strides 2,8 --fusion concat \
  --save experiments/mimic_MS_2-8_seed1