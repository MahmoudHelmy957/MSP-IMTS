#!/usr/bin/env bash
#SBATCH --job-name=physio_per_dim_ovrlp_2_4_8
#SBATCH --partition=NGPU
#SBATCH --nodelist=gpu-200
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --chdir=/home/ouass/Test/MSP-IMTS/lazyfor/new

set -euo pipefail
source "$HOME/venv310/bin/activate"
#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
#export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH="/home/ouass/Test/MSP-IMTS:/home/ouass/Test/MSP-IMTS/tPatchGNN:${PYTHONPATH-}"

cd /home/ouass/Test/MSP-IMTS/tPatchGNN

GPU=0
EPOCHS=600
PATIENCE=25
BATCH=32
LR=1e-3
HISTORY=24
QUANT=1.0

SCALES="2,4,8"
STRIDES="1,2,4"
METRIC="per_dim"   # or "per_dim"

for seed in {1..5}; do
  echo "=== Seed $seed ==="

  python run_models.py \
    --dataset physionet \
    --history $HISTORY \
    --quantization $QUANT \
    --hid_dim 64 \
    --te_dim 10 \
    --node_dim 10 \
    --nlayer 1 \
    --tf_layer 1 \
    --nhead 1 \
    --batch_size $BATCH \
    --lr $LR \
    --patience $PATIENCE \
    --epoch $EPOCHS \
    --seed $seed \
    --gpu $GPU \
    --metric "$METRIC" \
    --multi_scales "$SCALES" \
    --multi_strides "$STRIDES" \
    --fusion gated
done

# 'gated_feat', 'gated_feat_wconcat'