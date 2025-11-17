#!/usr/bin/env bash
#SBATCH --job-name=activity_single_mixer
#SBATCH --partition=TEST
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --chdir=/home/solgi/MSP-IMTS/logs

set -euo pipefail

# --- Environment setup ---
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate solgiland

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH="/home/solgi/MSP-IMTS:/home/solgi/MSP-IMTS/tPatchGNN:${PYTHONPATH-}"

cd /home/solgi/MSP-IMTS/tPatchGNN

# --- Diagnostics ---
echo "============================================================"
echo " SYSTEM & SOFTWARE INFO"
date
hostname
nvidia-smi
python -c "import torch; print('Torch:', torch.__version__, 'CUDA:', torch.version.cuda, 'cuDNN:', torch.backends.cudnn.version()); print('Device:', torch.cuda.get_device_name(0))"
echo "============================================================"

# === Experiment configuration ===
SEED=5
GPU=0
EPOCHS=600
PATIENCE=80
BATCH=32
LR=1e-3
HISTORY=4000

echo "============================================================"
echo "  Activity Single-Scale + NodeMixer experiment"
echo "  SEED=$SEED"
echo "  GPU=$GPU"
echo "  HISTORY=$HISTORY"
echo "============================================================"

# --- Launch training ---
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

echo "============================================================"
echo "  Job finished successfully."
echo "  Logs are in: /home/solgi/MSP-IMTS/logs/"
echo "============================================================"
