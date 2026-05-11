# Unified-interactive PEBBLE launcher (variant of scripts/lunar_lander/500/web).
#
# Same params as the split-workflow web script, but adds:
#   exit_after_max_feedback_checkpoint=false
# so the run saves a full checkpoint at max_feedback (for offline RL replay /
# comparison) but KEEPS TRAINING all the way to num_train_steps. One job does
# both label collection and the pure-RL phase.

# for seed in 12345 23451 78906 89067 6789; do
for seed in 12345; do
  PARENT_RUN_DIR="$SCRATCH/compare_runs/pebble/lunarlander_web_full/${SLURM_JOB_ID}"
  export COMPARE_RUN_DIR="$PARENT_RUN_DIR/seed_${seed}"
    mkdir -p "$COMPARE_RUN_DIR"
    echo "COMPARE_RUN_DIR=$COMPARE_RUN_DIR"

    echo ""
    echo "============================================================"
    echo "  ONLINE WEB LABELING MODE (UNIFIED — labels + full RL in one job)"
    echo "  On the login node (VS Code terminal), run ONCE for all seeds:"
    echo ""
    echo "    OPENBLAS_NUM_THREADS=1 python label_web.py --mode online \\"
    echo "      --online_query_dir $PARENT_RUN_DIR \\"
    echo "      --output $PARENT_RUN_DIR/human_labels.pkl \\"
    echo "      --port 8888"
    echo ""
    echo "  Forward port 8888 in VS Code, open http://localhost:8888/online"
    echo ""
    echo "  This job will save a checkpoint at max_feedback for later offline"
    echo "  replay/comparison, then continue training to num_train_steps."
    echo "============================================================"
    echo ""

    python -u train_PEBBLE.py \
      env=gym_LunarLanderContinuous-v2 seed=$seed \
      device=cuda \
      max_episode_steps=400 \
      agent.params.actor_lr=0.0005 agent.params.critic_lr=0.0005 \
      gradient_update=1 activation=tanh \
      num_unsup_steps=9000 num_train_steps=1000000 \
      num_interact=20000 max_feedback=100 \
      reward_batch=25 reward_update=25 \
      segment=50 \
      feed_type=7 \
      save_video=false \
      save_query_videos=false \
      checkpoint_at_max_feedback=true \
      exit_after_max_feedback_checkpoint=false \
      teacher_beta=-1 teacher_gamma=1 \
      teacher_eps_mistake=0 teacher_eps_skip=0 teacher_eps_equal=0

done
