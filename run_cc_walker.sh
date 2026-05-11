#!/bin/bash
#SBATCH --job-name=pebble
#SBATCH --account=aip-mtaylor3
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/walkerwalk/%x_%j.out
#SBATCH --error=logs/walkerwalk/%x_%j.err


# 1. Clean environment
module --force purge
module load StdEnv/2023

# 2. Activate conda properly
#source /scratch/marzii/miniconda3/etc/profile.d/conda.sh
#conda activate bpref39
source ~/activate_bpref39.sh

# export COMPARE_RUN_DIR="$SCRATCH/compare_runs/pebble/${SLURM_JOB_ID}"
# mkdir -p "$COMPARE_RUN_DIR"
# echo "COMPARE_RUN_DIR=$COMPARE_RUN_DIR"

# 4. Headless MuJoCo
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
unset DISPLAY
export PYTHONWARNINGS="ignore::DeprecationWarning"

python - <<'PY'
import torch
print("torch", torch.__version__, "compiled_cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0), "cap", torch.cuda.get_device_capability(0))
PY

echo "starting training..."
echo "Starting task $SLURM_ARRAY_TASK_ID"
echo "hostname=$(hostname)"
echo "python=$(which python)"
echo "jobid=$SLURM_JOB_ID"

start_time=`date +%s`

# 5. Run
cd ~/BPref2 || exit 1
#python train_PEBBLE.py env=walker_walk seed=12345 device=cpu
./scripts/walker_walk/500/oracle/run_PEBBLE.sh

end_time=`date +%s`
echo "run time $((end_time-start_time)) sec"
