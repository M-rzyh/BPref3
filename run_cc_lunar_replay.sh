#!/bin/bash
#SBATCH --job-name=pebble-replay
#SBATCH --account=aip-mtaylor3
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/lunarlander/%x_%j.out
#SBATCH --error=logs/lunarlander/%x_%j.err

# Replay a crashed web-mode run from saved labels.
# Reads metadata.pkl + response.pkl from REPLAY_DIR, applies them directly,
# then auto-checkpoints when max_feedback is hit and exits.
#
# Usage:
#   REPLAY_DIR=/scratch/marzii/compare_runs/pebble/lunarlander_web/<JOBID>/seed_<SEED>/online_queries \
#     sbatch run_cc_lunar_replay.sh

set -e

if [ -z "${REPLAY_DIR:-}" ]; then
  echo "ERROR: set REPLAY_DIR=/path/to/online_queries"
  exit 1
fi

if [ ! -d "$REPLAY_DIR" ]; then
  echo "ERROR: REPLAY_DIR does not exist: $REPLAY_DIR"
  exit 1
fi

# Pull seed from any batch's metadata if available, else default 12345
SEED=${SEED:-12345}

# By default the new run writes to .../recovered_<JOBID>/seed_<SEED>
# (same env-name root, fresh JOBID so old run is untouched)
PARENT_RUN_DIR=${PARENT_RUN_DIR:-"$SCRATCH/compare_runs/pebble/lunarlander_web/recovered_${SLURM_JOB_ID}"}
export COMPARE_RUN_DIR="$PARENT_RUN_DIR/seed_${SEED}"
mkdir -p "$COMPARE_RUN_DIR"

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

echo "===================================================="
echo "REPLAY MODE"
echo "REPLAY_DIR=$REPLAY_DIR"
echo "SEED=$SEED"
echo "OUTPUT=$COMPARE_RUN_DIR"
echo "JOBID=$SLURM_JOB_ID"
echo "===================================================="

start_time=$(date +%s)

cd ~/BPref3 || exit 1

# -u: unbuffered stdout/stderr so SLURM .out updates live (helps diagnose hangs)
python -u train_PEBBLE.py \
  env=gym_LunarLanderContinuous-v2 seed=$SEED \
  device=cuda max_episode_steps=400 \
  agent.params.actor_lr=0.0005 agent.params.critic_lr=0.0005 \
  gradient_update=1 activation=tanh \
  num_unsup_steps=9000 num_train_steps=1000000 \
  num_interact=20000 max_feedback=100 \
  reward_batch=25 reward_update=25 segment=50 \
  feed_type=7 \
  save_video=false save_query_videos=false \
  checkpoint_at_max_feedback=true \
  web_replay_from="$REPLAY_DIR" \
  teacher_beta=-1 teacher_gamma=1 \
  teacher_eps_mistake=0 teacher_eps_skip=0 teacher_eps_equal=0

end_time=$(date +%s)
echo "run time $((end_time-start_time)) sec"
echo ""
echo "Checkpoint at: $COMPARE_RUN_DIR/pebble/checkpoint.pt"
echo "To finish pure RL:"
echo "  RESUME_FROM=$COMPARE_RUN_DIR/pebble/checkpoint.pt sbatch run_cc_lunar_resume.sh"
