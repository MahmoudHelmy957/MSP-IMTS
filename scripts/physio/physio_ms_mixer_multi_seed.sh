#!/usr/bin/env bash
#SBATCH --job-name=physio_STUD_ms_sa_multi
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

# Deterministic CuBLAS config
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# Reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
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

# ==== Common config ====
GPU=0
EPOCHS=600
PATIENCE=12
BATCH=16          # halved to avoid OOM
LR=1e-3
HISTORY=24
QUANT=1.0
SCALES="2,8"
STRIDES="2,8"
FUSION="concat"
HID=32

echo "============================================================"
echo "  PhysioNet multi-scale + concat fusion (multi-seed)"
echo "  HISTORY=$HISTORY  SCALES=$SCALES  STRIDES=$STRIDES  HID_DIM=$HID  BATCH=$BATCH"
echo "============================================================"

# ==== Loop over seeds ====
for SEED in 1 2 3 4 5; do
  echo ">>> Running SEED=$SEED"
  LOGFILE="/home/solgi/MSP-IMTS/logs/physio_STUD_ms_sa_seed${SEED}_$(date +'%Y%m%d_%H%M%S').log"

  python run_models.py \
    --dataset physionet \
    --history $HISTORY \
    --quantization $QUANT \
    --hid_dim $HID \
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

  echo ">>> Finished SEED=$SEED; log saved to $LOGFILE"
done

echo "============================================================"
echo "  All seeds completed successfully."
echo "  Logs are in: /home/solgi/MSP-IMTS/logs/"
echo "============================================================"
