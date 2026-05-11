#!/bin/bash
#SBATCH --job-name=pebble-savevids
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

echo "starting training with SAVE_QUERY_VIDEOS=true..."
echo "hostname=$(hostname)"
echo "python=$(which python)"
echo "jobid=$SLURM_JOB_ID"

start_time=`date +%s`

# 4. Run with query video saving enabled for offline human labeling
cd ~/BPref3 || exit 1
export SAVE_QUERY_VIDEOS=true
./scripts/lunar_lander/500/oracle/run_PEBBLE.sh

end_time=`date +%s`
echo "run time $((end_time-start_time)) sec"
echo ""
echo "Query videos saved to: $COMPARE_RUN_DIR/query_videos/"
echo "To label offline:"
echo "  python human_label.py --query_dir \$COMPARE_RUN_DIR/query_videos --output human_labels.pkl"

# Script: run_cc_lunar_savevids.sh                                            
#   Purpose: Same training BUT saves segment query videos for offline labeling  
#   When to use: sbatch run_cc_lunar_savevids.sh — run this first to generate videos you'll label later 