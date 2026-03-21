#!/usr/bin/env bash
#SBATCH --job-name=physio_ss_global
#SBATCH --partition=NGPU
#SBATCH --nodelist=gpu-200
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --chdir=/home/ouass/Test/MSP-IMTS/lazyfor

set -euo pipefail
source /home/ouass/venv310/bin/activate
#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
#export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH="/home/ouass/Test/MSP-IMTS:/home/ouass/Test/MSP-IMTS/tPatchGNN:${PYTHONPATH-}"

cd /home/ouass/Test/MSP-IMTS/tPatchGNN

patience=10
gpu=0
numm=8
METRIC="global"   # or "per_dim"

for seed in {1..5}; do
  echo "=== Seed $seed ==="
  python run_models.py \
    --dataset physionet --state 'def' --history 24 \
    --patience $patience --batch_size 32 --lr 1e-3 \
    --patch_size $numm --stride $numm --nhead 1 --tf_layer 1 --nlayer 1 \
    --te_dim 10 --node_dim 10 --hid_dim 32 --metric "$METRIC" \
    --outlayer Linear --seed $seed \
    --quantization 1.0
done
