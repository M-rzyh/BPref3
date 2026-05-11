# for seed in 12345 23451 78906 89067 6789; do
for seed in 12345; do
  # Per-seed subdir so each seed has its own online_queries/. label_web.py
  # walks the parent dir recursively to find batches across seeds.
  PARENT_RUN_DIR="$SCRATCH/compare_runs/pebble/lunarlander_web/${SLURM_JOB_ID}"
  export COMPARE_RUN_DIR="$PARENT_RUN_DIR/seed_${seed}"
    mkdir -p "$COMPARE_RUN_DIR"
    echo "COMPARE_RUN_DIR=$COMPARE_RUN_DIR"

    echo ""
    echo "============================================================"
    echo "  ONLINE WEB LABELING MODE"
    echo "  On the login node (VS Code terminal), run ONCE for all seeds:"
    echo ""
    echo "    OPENBLAS_NUM_THREADS=1 python label_web.py --mode online \\"
    echo "      --online_query_dir $PARENT_RUN_DIR \\"
    echo "      --output $PARENT_RUN_DIR/human_labels.pkl \\"
    echo "      --port 8888"
    echo ""
    echo "  (Pointed at the parent dir — label_web walks all seed_*/online_queries/)"
    echo "  Then forward port 8888 in VS Code and open http://localhost:8888/online"
    echo "============================================================"
    echo ""

    python train_PEBBLE.py \
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
      teacher_beta=-1 teacher_gamma=1 \
      teacher_eps_mistake=0 teacher_eps_skip=0 teacher_eps_equal=0

done
