#!/bin/bash
# TRUE-REWARD SAC BASELINE for the PEBBLE LunarLander study.
#
# Same SAC as inside PEBBLE (same config/agent/sac.yaml: actor/critic 1024x2 ReLU,
# squashed Gaussian, double Q, Adam, actor/critic lr 5e-4, alpha lr 1e-4, batch 1024,
# buffer 1e6, gamma 0.99, tau 0.005, actor update 1, critic target update 2,
# init_temperature 0.1, learnable temperature, target entropy -2), same environment
# and 1000-step episode limit, same 1M env-step budget, same seeds, 1 gradient step
# per env step.
#
# Differences from the SAC inside PEBBLE, all intentional:
#   - reward = TRUE environment reward (no reward model, no r_hat)
#   - no replay-buffer relabeling
#   - NO 9000-step unsupervised/intrinsic phase (num_unsup_steps=0), and therefore
#     no critic reset + 100 catch-up updates at step 10k
#
# One seed per job (SEEDS=<seed>). Outputs go to $SCRATCH, never $HOME.
# Final policy is saved by train_SAC.py as actor_<step>.pt / critic_<step>.pt /
# critic_target_<step>.pt in the run dir.
#
# Submit (from ~/BPref3):
#   for s in 12345 23451 78906 89067 6789 13571 24681 35791 46801 57911; do
#     sbatch --time=06:00:00 --job-name=sac-truereward-s${s} \
#       --export=ALL,RUN_SCRIPT=./scripts/lunar_lander/100/sac_truereward/run_SAC.sh,SEEDS=$s \
#       run_cc_lunar.sh
#   done

# Overridable per submission; defaults are the matched-baseline settings.
MAX_EP_STEPS=${MAX_EP_STEPS:-1000}
NUM_SEED_STEPS=${NUM_SEED_STEPS:-1000}      # PEBBLE's random-action warmup (train.yaml default is 5000)
NUM_UNSUP_STEPS=${NUM_UNSUP_STEPS:-0}       # 0 = no intrinsic pre-training phase
NUM_TRAIN_STEPS=${NUM_TRAIN_STEPS:-1000000}
ACTOR_LR=${ACTOR_LR:-0.0005}
CRITIC_LR=${CRITIC_LR:-0.0005}

for seed in ${SEEDS:-12345}; do
  OUT="$SCRATCH/compare_runs/sac_truereward/lunarlander/${SLURM_JOB_ID}/seed_${seed}"
  mkdir -p "$OUT"
  echo "OUT=$OUT"
  echo "settings: cap=$MAX_EP_STEPS seed_steps=$NUM_SEED_STEPS unsup=$NUM_UNSUP_STEPS steps=$NUM_TRAIN_STEPS lr=$ACTOR_LR/$CRITIC_LR seed=$seed"

  python train_SAC.py \
    env=gym_LunarLanderContinuous-v2 \
    seed=$seed \
    device=cuda \
    max_episode_steps=$MAX_EP_STEPS \
    num_seed_steps=$NUM_SEED_STEPS \
    num_unsup_steps=$NUM_UNSUP_STEPS \
    num_train_steps=$NUM_TRAIN_STEPS \
    agent.params.actor_lr=$ACTOR_LR \
    agent.params.critic_lr=$CRITIC_LR \
    hydra.run.dir="$OUT"
done
