#!/usr/bin/env bash
#SBATCH --job-name=ushcn_MS_grid
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30000
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --chdir=/home/ouass/Test/MSP-IMTS/logs/ushcn

set -euo pipefail
source "$HOME/venv310/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONPATH="$HOME/Test/t-PatchGNN:$HOME/Test/t-PatchGNN/tPatchGNN:${PYTHONPATH-}"
cd "$HOME/Test/t-PatchGNN/tPatchGNN"

gpu=0; patience=15; seed=1

declare -a SCALES=("1,2,3" "2,4,8" "3,6,12")
declare -a STRIDES=("1,1,1" "1,2,4" "")

for sc in "${SCALES[@]}"; do
  for st in "${STRIDES[@]}"; do
    tag="sc${sc//,/−}_st${st//,/−}"
    echo "=== GRID: scales=${sc} strides=${st:-same} ==="
    python run_models.py \
      --dataset ushcn --state 'def' \
      --history 24 --patience ${patience} --batch_size 32 --lr 1e-3 \
      --patch_size 24 --stride 24 --nhead 1 --tf_layer 1 --nlayer 1 \
      --te_dim 10 --node_dim 10 --hid_dim 64 \
      --outlayer Linear --seed ${seed} --gpu ${gpu} \
      --multi_scales "${sc}" \
      --multi_strides "${st}" \
      --fusion concat \
      --save experiments/ushcn_MS_${tag}_seed${seed}
  done
done
