#!/bin/bash
#SBATCH --job-name=bpref_pebble
#SBATCH --account=aip-mtaylor3
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# 1. Clean environment
module --force purge
module load StdEnv/2023

# 2. Activate conda properly
#source /scratch/marzii/miniconda3/etc/profile.d/conda.sh
#conda activate bpref39
source ~/activate_bpref39.sh

# 4. Headless MuJoCo
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
unset DISPLAY
export PYTHONWARNINGS="ignore::DeprecationWarning"

start_time=`date +%s`
echo "starting training..."
echo "Starting task $SLURM_ARRAY_TASK_ID"

# 5. Run
cd ~/BPref2 || exit 1
python train_PEBBLE.py env=walker_walk seed=12345 device=cpu
#./scripts/walker_walk/500/oracle/run_PEBBLE.sh

end_time=`date +%s`
runtime=$((end_time-start_time))

echo "run time"
echo $runtime
