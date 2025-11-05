#!/usr/bin/env bash
#SBATCH --job-name=mimic_SS_ps8_s8
#SBATCH --partition=TEST
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30000
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
  --patience 10 --epoch 1000 \
  --batch_size 16 --lr 1e-3 \
  --patch_size 8 --stride 8 \
  --nhead 1 --tf_layer 1 --nlayer 1 \
  --te_dim 10 --node_dim 10 --hid_dim 64 \
  --outlayer Linear --seed 1 --gpu 0 \
  --save experiments/mimic_SS_ps8_s8_seed1
