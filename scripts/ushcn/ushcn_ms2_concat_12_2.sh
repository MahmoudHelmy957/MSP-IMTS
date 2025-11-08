#!/usr/bin/env bash
#SBATCH --job-name=ushcn_MS_same
#SBATCH --partition=TEST
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30000
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --chdir=/home/ouass/Test/MSP-IMTS/logs/ushcn

set -euo pipefail
source $HOME/venv310/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH="/home/ouass/Test/MSP-IMTS:/home/ouass/Test/MSP-IMTS/tPatchGNN:${PYTHONPATH-}"

cd $HOME/Test/MSP-IMTS/tPatchGNN

python run_models.py \
  --dataset ushcn --state def --history 24 \
  --batch_size 192 --lr 5e-4 --w_decay 1e-5 \
  --patience 5 --epoch 1000 \
  --nhead 1 --tf_layer 1 --nlayer 1 \
  --te_dim 10 --node_dim 10 --hid_dim 32 \
  --outlayer Linear --seed 1 --gpu 0 \
  --multi_scales 1,2 --multi_strides 1,2 --fusion concat
