#!/usr/bin/env bash
#SBATCH --job-name=activity_single_mixer_seeds
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --array=1-6
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --chdir=/home/solgi/MSP-IMTS/logs

set -euo pipefail

# === SETUP ENVIRONMENT ===
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate solgiland

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH="/home/solgi/MSP-IMTS:/home/solgi/MSP-IMTS/tPatchGNN:${PYTHONPATH-}"

cd /home/solgi/MSP-IMTS/tPatchGNN

# ------------------------------
# SYSTEM INFO
# ------------------------------
echo "============================================================"
echo " SEED JOB: $SLURM_ARRAY_TASK_ID  (SLURM_ARRAY_JOB_ID=$SLURM_ARRAY_JOB_ID)"
echo "============================================================"
date
hostname
nvidia-smi
echo "============================================================"

# === CONFIG ===
SEED=$SLURM_ARRAY_TASK_ID
GPU=0
EPOCHS=600
PATIENCE=80
BATCH=32
LR=1e-3
HISTORY=3000

echo "== Running Single-Scale NodeMixer Experiment =="
echo " SEED = $SEED"
echo "================================================"

# === RUN TRAINING ===
python run_models.py \
  --dataset activity \
  --history ${HISTORY} \
  --hid_dim 32 \
  --te_dim 10 \
  --node_dim 10 \
  --nlayer 1 \
  --tf_layer 1 \
  --nhead 1 \
  --batch_size ${BATCH} \
  --lr ${LR} \
  --w_decay 1e-4 \
  --patience ${PATIENCE} \
  --epoch ${EPOCHS} \
  --seed ${SEED} \
  --gpu ${GPU} \
  --multi_scales "" \
  --fusion concat \
  --state activity_single_scale_mixer

echo "================================================"
echo " SEED $SEED completed."
echo " Logs saved to /home/solgi/MSP-IMTS/logs"
echo "================================================"
