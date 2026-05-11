# Resume PEBBLE training from a checkpoint produced by checkpoint_at_max_feedback.
# Required env var:  RESUME_FROM=/path/to/checkpoint.pt
#
# The resumed run uses the same work_dir (=parent of checkpoint), so logs and
# TensorBoard continue in the same files. No new queries are issued because
# total_feedback already equals max_feedback.

if [ -z "${RESUME_FROM:-}" ]; then
  echo "ERROR: set RESUME_FROM=/path/to/checkpoint.pt"
  exit 1
fi

# COMPARE_RUN_DIR is the parent of work_dir (work_dir = $COMPARE_RUN_DIR/pebble)
WORK_DIR=$(dirname "$RESUME_FROM")
export COMPARE_RUN_DIR=$(dirname "$WORK_DIR")
echo "RESUME_FROM=$RESUME_FROM"
echo "WORK_DIR=$WORK_DIR"
echo "COMPARE_RUN_DIR=$COMPARE_RUN_DIR"

# Pull seed from the checkpoint metadata (sidecar JSON written by save_checkpoint)
META="${RESUME_FROM}.meta.json"
if [ -f "$META" ]; then
  SEED=$(python -c "import json; print(json.load(open('$META'))['seed'])")
else
  SEED=12345
fi
echo "seed=$SEED"

# -u: unbuffered so SLURM .out updates live (helps diagnose hangs)
python -u train_PEBBLE.py \
  env=gym_LunarLanderContinuous-v2 seed=$SEED \
  device=cuda \
  max_episode_steps=400 \
  agent.params.actor_lr=0.0005 agent.params.critic_lr=0.0005 \
  gradient_update=1 activation=tanh \
  num_unsup_steps=9000 num_train_steps=1000000 \
  num_interact=20000 max_feedback=100 \
  reward_batch=25 reward_update=25 \
  segment=50 \
  feed_type=7 \
  save_video=true \
  save_query_videos=false \
  checkpoint_at_max_feedback=false \
  resume_from="$RESUME_FROM" \
  teacher_beta=-1 teacher_gamma=1 \
  teacher_eps_mistake=0 teacher_eps_skip=0 teacher_eps_equal=0
