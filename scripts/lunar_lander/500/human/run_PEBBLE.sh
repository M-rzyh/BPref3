HUMAN_LABELS=${HUMAN_LABELS_PATH:-/scratch/marzii/compare_runs/pebble/lunarlander/4624956/human_labels.pkl}

for seed in 12345; do
  export COMPARE_RUN_DIR="$SCRATCH/compare_runs/pebble/lunarlander_human/${SLURM_JOB_ID}"
    mkdir -p "$COMPARE_RUN_DIR"
    echo "COMPARE_RUN_DIR=$COMPARE_RUN_DIR"
    echo "HUMAN_LABELS=$HUMAN_LABELS"

    python train_PEBBLE.py env=gym_LunarLanderContinuous-v2 seed=$seed feed_type=0 num_train_steps=1000000 --cfg job | egrep "^(env|seed|feed_type|device|num_train_steps|max_feedback|reward_batch|reward_update|num_interact|num_unsup_steps|human_labels_path):"

    python train_PEBBLE.py \
      env=gym_LunarLanderContinuous-v2 seed=$seed \
      device=cuda \
      max_episode_steps=400 \
      agent.params.actor_lr=0.0005 agent.params.critic_lr=0.0005 \
      gradient_update=1 activation=tanh \
      num_unsup_steps=9000 num_train_steps=1000000 \
      num_interact=20000 max_feedback=1000 \
      reward_batch=25 reward_update=25 \
      feed_type=0 \
      human_labels_path=$HUMAN_LABELS \
      save_video=${SAVE_VIDEO:-true} \
      teacher_beta=-1 teacher_gamma=1 teacher_eps_mistake=0 teacher_eps_skip=0 teacher_eps_equal=0

done
