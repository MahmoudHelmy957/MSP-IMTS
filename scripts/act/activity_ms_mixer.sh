#!/usr/bin/env bash
#SBATCH --job-name=activity_ms_mixer_multi
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=10:00:00
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

# --- Hardware / software diagnostics ---
echo "============================================================"
echo " SYSTEM & SOFTWARE INFO"
date
hostname
nvidia-smi
python -c "import torch; print('Torch:', torch.__version__, 'CUDA:', torch.version.cuda, 'cuDNN:', torch.backends.cudnn.version()); print('Device:', torch.cuda.get_device_name(0))"
echo "============================================================"

# === Common hyperparameters ===
GPU=0
EPOCHS=600
PATIENCE=60
BATCH=32
LR=1e-3
HISTORY=3000
SCALES="200,1200"
STRIDES="100,600"

echo "============================================================"
echo "  Activity multi-scale + NodeMixer fusion (multi-seed)"
echo "  HISTORY=$HISTORY  SCALES=$SCALES  STRIDES=$STRIDES"
echo "============================================================"

# --- Loop over seeds ---
for SEED in 1 2 3 4 5; do
  echo ">>> Running SEED=$SEED"
  LOGFILE="/home/solgi/MSP-IMTS/logs/activity_ms_mixer_seed${SEED}_$(date +'%Y%m%d_%H%M%S').log"

  python run_models.py \
    --dataset activity \
    --history $HISTORY \
    --hid_dim 32 \
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
    --fusion concat \
    --state activity_mixer \
    > "$LOGFILE" 2>&1

  echo ">>> Finished SEED=$SEED; log saved to $LOGFILE"
done

echo "============================================================"
echo "  All seeds completed successfully."
echo "  Logs are in: /home/solgi/MSP-IMTS/logs/"
echo "============================================================"
