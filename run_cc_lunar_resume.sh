#!/bin/bash
#SBATCH --job-name=pebble-resume
#SBATCH --account=aip-mtaylor3
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:30:00
#SBATCH --output=logs/lunarlander/%x_%j.out
#SBATCH --error=logs/lunarlander/%x_%j.err

# Resume the pure-RL portion of a PEBBLE run from a checkpoint.
#
# Usage:
#   RESUME_FROM=/scratch/.../seed_12345/pebble/checkpoint.pt sbatch run_cc_lunar_resume.sh

set -e

if [ -z "${RESUME_FROM:-}" ]; then
  echo "ERROR: set RESUME_FROM=/path/to/checkpoint.pt before sbatch"
  exit 1
fi

module --force purge
module load StdEnv/2023

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

echo "RESUMING from $RESUME_FROM"
echo "hostname=$(hostname)"
echo "jobid=$SLURM_JOB_ID"

start_time=$(date +%s)

cd ~/BPref3 || exit 1
RESUME_FROM="$RESUME_FROM" ./scripts/lunar_lander/500/resume/run_PEBBLE.sh

end_time=$(date +%s)
echo "run time $((end_time-start_time)) sec"
