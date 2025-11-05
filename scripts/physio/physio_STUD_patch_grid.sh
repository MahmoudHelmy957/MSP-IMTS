#!/usr/bin/env bash
#SBATCH --job-name=physio_STUD_patch_grid
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-9
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --chdir=/home/ouass/Test/t-PatchGNN/logs

set -euo pipefail
source $HOME/venv310/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH="$HOME/Test/t-PatchGNN:$HOME/Test/t-PatchGNN/tPatchGNN:${PYTHONPATH-}"

cd $HOME/Test/t-PatchGNN/tPatchGNN

# idx 0..4 => (ps=8, s=4) seeds 1..5
# idx 5..9 => (ps=12, s=6) seeds 1..5
CFG_PS=(8 8 8 8 8  12 12 12 12 12)
CFG_ST=(4 4 4 4 4   6  6  6  6  6)

idx=${SLURM_ARRAY_TASK_ID}
ps=${CFG_PS[$idx]}
st=${CFG_ST[$idx]}
seed=$(( (idx % 5) + 1 ))

echo "Running STUD: ps=$ps stride=$st seed=$seed"

python run_models.py \
  --dataset physionet --state def --history 24 \
  --patience 10 --batch_size 32 --lr 1e-3 \
  --patch_size $ps --stride $st --nhead 1 --tf_layer 1 --nlayer 1 \
  --te_dim 10 --node_dim 10 --hid_dim 64 \
  --outlayer Linear --seed $seed --gpu 0 \
  --quantization 1.0 \
  --save experiments/physio_ps${ps}_s${st}_seed${seed}
