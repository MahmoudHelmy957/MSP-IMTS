#!/usr/bin/env bash
#SBATCH --job-name=ushcn_MS_2sc_6-12
#SBATCH --partition=TEST
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30000
#SBATCH --time=00:59:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --chdir=/home/ouass/Test/MSP-IMTS/logs/ushcn

set -euo pipefail
source "$HOME/venv310/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONPATH="$HOME/Test/t-PatchGNN:$HOME/Test/t-PatchGNN/tPatchGNN:${PYTHONPATH-}"
cd "$HOME/Test/t-PatchGNN/tPatchGNN"

gpu=0; patience=20; seed=1
python run_models.py \
  --dataset ushcn --state 'def' --history 24 --patience ${patience} \
  --batch_size 32 --lr 1e-3 --patch_size 24 --stride 24 \
  --nhead 1 --tf_layer 1 --nlayer 1 --te_dim 10 --node_dim 10 --hid_dim 64 \
  --outlayer Linear --seed ${seed} --gpu ${gpu} \
  --multi_scales "6,12" --fusion concat \
  --save experiments/ushcn_MS_2sc_6-12_seed${seed}
