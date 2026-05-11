#!/bin/bash
#SBATCH --job-name=pebble-human
#SBATCH --account=aip-mtaylor3
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=04:30:00
#SBATCH --output=logs/lunarlander/%x_%j.out
#SBATCH --error=logs/lunarlander/%x_%j.err


# 1. Clean environment
module --force purge
module load StdEnv/2023

# 2. Activate conda (use bpref39_clone, not bpref39)
eval "$(/scratch/marzii/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch/marzii/envs/bpref39_clone
export PATH=/scratch/marzii/envs/bpref39_clone/bin:$PATH
hash -r
echo "Activated bpref39_clone"
which python
python -V

# 3. No MuJoCo needed for LunarLander, but suppress noisy warnings
export PYTHONWARNINGS="ignore::DeprecationWarning"

# Keep LunarLander SLURM logs in a dedicated subdirectory.
mkdir -p logs/lunarlander

python - <<'PY'
import torch
print("torch", torch.__version__, "compiled_cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0), "cap", torch.cuda.get_device_capability(0))
PY

echo "starting training with HUMAN LABELS..."
echo "hostname=$(hostname)"
echo "python=$(which python)"
echo "jobid=$SLURM_JOB_ID"

# Path to human labels (change this path to use different labels)
export HUMAN_LABELS_PATH=${HUMAN_LABELS_PATH:-/scratch/marzii/compare_runs/pebble/lunarlander/4624956/human_labels.pkl}

start_time=`date +%s`

# 4. Run with offline human labels
cd ~/BPref3 || exit 1
./scripts/lunar_lander/500/human/run_PEBBLE.sh

end_time=`date +%s`
echo "run time $((end_time-start_time)) sec"

# Script: run_cc_lunar_human.sh
#   Purpose: Trains PEBBLE using pre-saved human labels (from
#     scripts/lunar_lander/500/human/run_PEBBLE.sh)
#   When to use: sbatch run_cc_lunar_human.sh — run AFTER you've labeled the
#     videos from

# How to use: at run time: HUMAN_LABELS_PATH=/path/to/labels.pkl sbatch run_cc_lunar_human.sh
#     or just change the default HUMAN_LABELS_PATH in the script to point to your labels.pkl file.