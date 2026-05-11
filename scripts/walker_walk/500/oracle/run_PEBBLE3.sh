# FEED_TYPE=${1:-0}

# # for seed in 12345 23451 78906 89067 6789; do
# for seed in 12345; do
#     export COMPARE_RUN_DIR="$SCRATCH/compare_runs/pebble/${SLURM_JOB_ID}/seed_${seed}"
#     mkdir -p "$COMPARE_RUN_DIR"
#     echo "COMPARE_RUN_DIR=$COMPARE_RUN_DIR"
#     python train_PEBBLE.py  env=walker_walk seed=$seed agent.params.actor_lr=0.0005 agent.params.critic_lr=0.0005 gradient_update=1 activation=tanh num_unsup_steps=9000 num_train_steps=1000000 num_interact=20000 max_feedback=1000 reward_batch=50 reward_update=50 feed_type=$FEED_TYPE teacher_beta=-1 teacher_gamma=1 teacher_eps_mistake=0 teacher_eps_skip=0 teacher_eps_equal=0
# done


FEED_TYPE=${1:-0}

# for seed in 23451 78906 89067 6789; do
for seed in 12345; do
    export COMPARE_RUN_DIR="$SCRATCH/compare_runs/pebble/${SLURM_JOB_ID}/seed_${seed}"
    mkdir -p "$COMPARE_RUN_DIR"
    echo "COMPARE_RUN_DIR=$COMPARE_RUN_DIR"

    # print effective config (super useful for debugging)
    python train_PEBBLE.py env=walker_walk seed=$seed feed_type=$FEED_TYPE num_train_steps=500000 --cfg job | egrep "^(env|seed|feed_type|device|num_train_steps|max_feedback|reward_batch|reward_update|num_interact|num_unsup_steps):"

    python train_PEBBLE.py \
      env=walker_walk seed=$seed \
      device=cuda \
      agent.params.actor_lr=0.0005 agent.params.critic_lr=0.0005 \
      gradient_update=1 activation=tanh \
      num_unsup_steps=9000 num_train_steps=1000000\
      num_interact=20000 max_feedback=10 \
      reward_batch=50 reward_update=50 \
      feed_type=$FEED_TYPE \
      teacher_beta=-1 teacher_gamma=1 teacher_eps_mistake=0 teacher_eps_skip=0 teacher_eps_equal=0

    
done


#README:
# num_unsup_steps: number of unsupervised pretraining steps (PEBBLE only)
# num_train_steps: total number of training steps (including unsupervised pretraining)
