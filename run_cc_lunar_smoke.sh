#!/bin/bash
#SBATCH --job-name=pebble-smoke
#SBATCH --account=aip-mtaylor3
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/lunarlander/%x_%j.out
#SBATCH --error=logs/lunarlander/%x_%j.err

# End-to-end smoke for save-and-continue + resume parity.
#
# Phase A: tiny unified run with checkpoint_at_max_feedback=true and
#          exit_after_max_feedback_checkpoint=false. Trains 0 -> 1000 steps,
#          hits max_feedback=4 at step ~500, writes checkpoint, KEEPS training
#          to step 1000.
# Phase B: copies the checkpoint to a sibling dir, resumes from it. Trains
#          ~500 -> 1000 again as a separate job-equivalent.
# Phase C: validates that both phases produced the expected files.
#
# Total wall: ~3-5 min. sbatch wall set to 30 min for headroom.

set -e

module --force purge
module load StdEnv/2023
eval "$(/scratch/marzii/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch/marzii/envs/bpref39_clone
export PATH=/scratch/marzii/envs/bpref39_clone/bin:$PATH
hash -r
export PYTHONWARNINGS="ignore::DeprecationWarning"

mkdir -p logs/lunarlander
cd ~/BPref3

SEED=12345
SMOKE_ROOT="$SCRATCH/compare_runs/pebble/lunarlander_smoke/${SLURM_JOB_ID}"
UNIFIED_DIR="$SMOKE_ROOT/unified/seed_${SEED}"
COMPARE_DIR="$SMOKE_ROOT/compare/seed_${SEED}"

mkdir -p "$UNIFIED_DIR" "$COMPARE_DIR/pebble"

echo "=========================================="
echo " PHASE A: unified run (save AND continue) "
echo "=========================================="
echo "out -> $UNIFIED_DIR/pebble"

export COMPARE_RUN_DIR="$UNIFIED_DIR"

# Common smoke overrides; reused for both phases except the checkpoint flag.
COMMON=(
  env=gym_LunarLanderContinuous-v2 seed=$SEED
  device=cuda
  max_episode_steps=200
  agent.params.actor_lr=0.0005 agent.params.critic_lr=0.0005
  gradient_update=1 activation=tanh
  num_seed_steps=100
  num_unsup_steps=200
  num_train_steps=1000
  num_interact=200
  max_feedback=4
  reward_batch=2
  reward_update=10
  segment=20
  feed_type=0
  save_video=false
  save_query_videos=false
  teacher_beta=-1 teacher_gamma=1
  teacher_eps_mistake=0 teacher_eps_skip=0 teacher_eps_equal=0
)

python -u train_PEBBLE.py \
  "${COMMON[@]}" \
  checkpoint_at_max_feedback=true \
  exit_after_max_feedback_checkpoint=false

UNIFIED_CKPT="$UNIFIED_DIR/pebble/checkpoint.pt"
if [ ! -f "$UNIFIED_CKPT" ]; then
  echo "FAIL: no checkpoint.pt at $UNIFIED_CKPT (save-and-continue did not write checkpoint)"
  exit 2
fi
echo "[smoke] phase A produced checkpoint at $UNIFIED_CKPT"

echo
echo "=========================================="
echo " PHASE B: resume from copied checkpoint   "
echo "=========================================="

# Copy checkpoint + sidecars to a separate dir so the resume job's logs/csvs
# don't collide with the unified run's already-finalized files.
cp -p "$UNIFIED_DIR/pebble/checkpoint.pt"* "$COMPARE_DIR/pebble/"
echo "out -> $COMPARE_DIR/pebble"

# Unset COMPARE_RUN_DIR; resume derives work_dir from the checkpoint's parent.
unset COMPARE_RUN_DIR

python -u train_PEBBLE.py \
  "${COMMON[@]}" \
  checkpoint_at_max_feedback=false \
  exit_after_max_feedback_checkpoint=true \
  resume_from="$COMPARE_DIR/pebble/checkpoint.pt"

echo
echo "=========================================="
echo "          PHASE C: validation             "
echo "=========================================="

python -u - <<PYEOF
import json, os, sys
import pandas as pd

unified = "$UNIFIED_DIR/pebble"
compare = "$COMPARE_DIR/pebble"
fails = []

def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        fails.append(msg)

# 1. Unified checkpoint files
for fname in ('checkpoint.pt', 'checkpoint.pt.replay.npz', 'checkpoint.pt.meta.json'):
    p = os.path.join(unified, fname)
    check(os.path.exists(p) and os.path.getsize(p) > 0,
          f"unified/{fname} exists and non-empty")

# 2. Unified compute_time.json shows pure_rl > 0
ct_u = json.load(open(os.path.join(unified, 'compute_time.json')))
check(ct_u.get('phase_pure_rl_sec', 0) > 0,
      f"unified pure_rl phase ran (phase_pure_rl_sec={ct_u.get('phase_pure_rl_sec', 0):.2f}s, > 0)")

# 3. Unified train.csv reaches num_train_steps
udf = pd.read_csv(os.path.join(unified, 'train.csv'))
last_u = int(udf['step'].iloc[-1])
check(last_u >= 800,
      f"unified train.csv last_step={last_u} (>= 800 means full run finished)")

# 4. Compare train.csv exists, picks up after pretrain, reaches end
cdf = pd.read_csv(os.path.join(compare, 'train.csv'))
first_c = int(cdf['step'].iloc[0])
last_c = int(cdf['step'].iloc[-1])
check(first_c >= 400 and last_c >= 800,
      f"compare train.csv step range=[{first_c},{last_c}] (>= [400,800] means resume worked)")

# 5. Final actor saved in both
def has_actor(d):
    return any(f.startswith('actor_') and f.endswith('.pt') for f in os.listdir(d))
check(has_actor(unified), "unified saved final actor_*.pt")
check(has_actor(compare), "compare saved final actor_*.pt")

# 6. Compare compute_time.json exists and has phase_pure_rl_sec > 0
ct_c = json.load(open(os.path.join(compare, 'compute_time.json')))
check(ct_c.get('phase_pure_rl_sec', 0) > 0,
      f"compare pure_rl phase ran (phase_pure_rl_sec={ct_c.get('phase_pure_rl_sec', 0):.2f}s, > 0)")

print()
if fails:
    print(f"SMOKE FAIL ({len(fails)} checks failed)")
    for f in fails:
        print(f"  - {f}")
    sys.exit(3)
print("SMOKE PASS")
PYEOF

echo
echo "smoke complete."
echo "  unified: $UNIFIED_DIR"
echo "  compare: $COMPARE_DIR"
