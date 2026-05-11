#!/bin/bash
#SBATCH --job-name=pebble-web-full
#SBATCH --account=aip-mtaylor3
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=05:30:00
#SBATCH --output=logs/lunarlander/%x_%j.out
#SBATCH --error=logs/lunarlander/%x_%j.err

# Unified-interactive PEBBLE: collects 100 human labels via web AND runs the
# full pure-RL phase to num_train_steps in a single job. Saves a checkpoint at
# max_feedback so the same labels can also be replayed offline for comparison.
#
# 16G mem (vs 8G for split-workflow web) because the run_cc_lunar_resume job
# observed peaks of ~13 GB during the checkpoint write + load cycle.
# 5:30 wall time covers ~30 min labeling + ~4h45m RL + margin.

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

export PYTHONWARNINGS="ignore::DeprecationWarning"

mkdir -p logs/lunarlander

python - <<'PY'
import torch
print("torch", torch.__version__, "compiled_cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0), "cap", torch.cuda.get_device_capability(0))
PY

echo "starting UNIFIED INTERACTIVE WEB training (labels + RL in one job)..."
echo "hostname=$(hostname)"
echo "python=$(which python)"
echo "jobid=$SLURM_JOB_ID"

start_time=$(date +%s)

# 4. Run
cd ~/BPref3 || exit 1
./scripts/lunar_lander/500/web_full/run_PEBBLE.sh

end_time=$(date +%s)
echo "run time $((end_time-start_time)) sec"

# Script: run_cc_lunar_web_full.sh
#   Purpose: Trains PEBBLE with online web labeling AND finishes pure-RL phase
#            in a single job. Also writes a max_feedback checkpoint mid-run.
#   How to use:
#     1. sbatch run_cc_lunar_web_full.sh
#     2. Check the job log for the label_web.py command to run on login node
#     3. Run that command in a VS Code terminal
#     4. Forward port 8888, open http://localhost:8888/online
#     5. Label queries (~30 min). Job continues to num_train_steps automatically.
