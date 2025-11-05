#!/usr/bin/env bash
#SBATCH --job-name=act_MS_A
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --chdir=/home/ouass/Test/MSP-IMTS/act

set -euo pipefail
source "$HOME/venv310/bin/activate"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH="/home/ouass/Test/MSP-IMTS:/home/ouass/Test/MSP-IMTS/tPatchGNN:${PYTHONPATH-}"

cd /home/ouass/Test/MSP-IMTS/tPatchGNN

GPU=0
EPOCHS=400
PATIENCE=60
BATCH=64
LR=1e-3
WDECAY=1e-4
HISTORY=4000
HID=64
TE=10
ND=10
NL=1
TFL=1
NHEAD=1

# A-config (best so far)
SCALES="200,1200"
STRIDES="100,600"

for SEED in 1 2 3 4 5; do
  echo "======================================================="
  echo "Running ACTIVITY MS (concat) A: seed=$SEED  scales=$SCALES  strides=$STRIDES"
  echo "HISTORY=$HISTORY  HID=$HID  BATCH=$BATCH  LR=$LR  WDECAY=$WDECAY  EPOCHS=$EPOCHS  PATIENCE=$PATIENCE"
  echo "Start time: $(date '+%F %T')"
  echo "-------------------------------------------------------"

  python run_models.py \
    --dataset activity \
    --history "$HISTORY" \
    --hid_dim "$HID" \
    --te_dim "$TE" \
    --node_dim "$ND" \
    --nlayer "$NL" \
    --tf_layer "$TFL" \
    --nhead "$NHEAD" \
    --batch_size "$BATCH" \
    --lr "$LR" \
    --w_decay "$WDECAY" \
    --patience "$PATIENCE" \
    --epoch "$EPOCHS" \
    --seed "$SEED" \
    --gpu "$GPU" \
    --multi_scales "$SCALES" \
    --multi_strides "$STRIDES" \
    --fusion "concat"

  echo "Finished seed $SEED at $(date '+%F %T')"
  echo "======================================================="
  echo
done
