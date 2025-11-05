#!/usr/bin/env bash
#SBATCH --job-name=ushcn_MS_3-6-12_allseeds
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

patience=20
gpu=0
scales="4,6,12"
strides=""          # empty ⇒ same-as-scales
fusion="concat"

for seed in {1..5}; do
  echo "=== USHCN MULTI-SCALE (${scales} mo) | seed=${seed} ==="
  python run_models.py \
    --dataset ushcn --state 'def' \
    --history 24 --patience ${patience} --batch_size 32 --lr 1e-3 \
    --patch_size 24 --stride 24 --nhead 1 --tf_layer 1 --nlayer 1 \
    --te_dim 10 --node_dim 10 --hid_dim 64 \
    --outlayer Linear --seed ${seed} --gpu ${gpu} \
    --multi_scales "${scales}" \
    --multi_strides "${strides}" \
    --fusion ${fusion} \
    --save experiments/ushcn_MS_sc${scales//,/−}_seed${seed}
done
