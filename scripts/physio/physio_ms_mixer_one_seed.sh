#!/usr/bin/env bash
#SBATCH --job-name=physio_STUD_ms_sa_seed1
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=10:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --chdir=/home/solgi/MSP-IMTS/logs

set -euo pipefail

# ==== Environment ====
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate solgiland

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH="/home/solgi/MSP-IMTS:/home/solgi/MSP-IMTS/tPatchGNN:${PYTHONPATH-}"

cd /home/solgi/MSP-IMTS/tPatchGNN

echo "============================================================"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
python -V
nvidia-smi
echo "============================================================"

# ==== Config ====
GPU=0
EPOCHS=600
PATIENCE=12
BATCH=32
LR=1e-3
HISTORY=24
QUANT=1.0
SCALES="2,8"
STRIDES="2,8"
FUSION="concat"
SEED=1

echo "============================================================"
echo "  PhysioNet multi-scale + scale-attn fusion (TEST)"
echo "  SEED=$SEED  HISTORY=$HISTORY  SCALES=$SCALES  STRIDES=$STRIDES  FUSION=$FUSION"
echo "============================================================"

LOGFILE="/home/solgi/MSP-IMTS/logs/physio_TEST_ms_sa_seed${SEED}_$(date +'%Y%m%d_%H%M%S').log"

python run_models.py \
  --dataset physionet \
  --history $HISTORY \
  --quantization $QUANT \
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
  --fusion "$FUSION" \
  > "$LOGFILE" 2>&1

echo "============================================================"
echo "  Job finished successfully. Log: $LOGFILE"
echo "============================================================"
