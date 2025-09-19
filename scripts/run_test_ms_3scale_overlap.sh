#!/usr/bin/env bash
#SBATCH --job-name=physio_TEST_ms_3scale_ol
#SBATCH --partition=TEST
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30000
#SBATCH --time=00:59:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --chdir=/home/ouass/Test/t-PatchGNN/logs

set -euo pipefail
source "$HOME/venv310/bin/activate"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH="$HOME/Test/t-PatchGNN:$HOME/Test/t-PatchGNN/tPatchGNN:${PYTHONPATH-}"
cd "$HOME/Test/t-PatchGNN/tPatchGNN"

echo "Host: $(hostname)"; echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"; python -V

SEED=0
GPU=0
EPOCHS=60          
PATIENCE=20
BATCH=16           
LR=1e-3
HISTORY=24
QUANT=1.0

# Multi-scale: 2/8/24h with overlap 1/4/12h
SCALES="2,8,24"
STRIDES="1,4,12"

echo "Running TEST MS 3-scale overlap: scales=$SCALES strides=$STRIDES seed=$SEED"

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
  --seed $SEED \
  --gpu $GPU \
  --multi_scales "$SCALES" \
  --multi_strides "$STRIDES" \
  --fusion concat
