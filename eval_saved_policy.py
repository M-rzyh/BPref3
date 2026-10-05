"""Deterministic post-training evaluation of a saved PEBBLE / SAC policy.

Loads actor_<STEP>.pt from a finished run directory and rolls EPISODES episodes
in the real environment with the deterministic policy (act(..., sample=False)),
scoring with the TRUE environment reward. Writes final_eval_<EPISODES>ep.json
next to the weights. Nothing in the run directory is modified otherwise.

This matches the protocol used for the GAIL / BC / PT policies:
  * 50 episodes,
  * deterministic action selection,
  * true environment reward,
  * fresh env at the same time limit, episode e seeded EVAL_SEED_BASE + e
    (default EVAL_SEED_BASE = training seed * 1000, the same convention the
    GAIL and BC eval blocks use).

train_PEBBLE.py's own evaluate() cannot serve this purpose: it fires only when
step % eval_frequency == 0 AND an episode ends on that exact step, so eval.csv
is nearly empty in every run.

Run from ~/BPref3 with the bpref39_clone env, e.g.:

  MODEL_DIR=/scratch/marzii/compare_runs/pebble/lunarlander/973283/seed_12345/pebble \
  STEP=1000000 EPISODES=50 \
  python eval_saved_policy.py env=gym_LunarLanderContinuous-v2 seed=12345 \
      device=cuda max_episode_steps=1000 \
      agent.params.actor_lr=0.0005 agent.params.critic_lr=0.0005
"""
import json
import os

import hydra
import numpy as np

import utils


@hydra.main(config_path='config/train_PEBBLE.yaml', strict=True)
def main(cfg):
    model_dir = os.environ['MODEL_DIR']
    step = os.environ.get('STEP', '1000000')
    episodes = int(os.environ.get('EPISODES', '50'))
    seed_base = int(os.environ.get('EVAL_SEED_BASE', str(int(cfg.seed) * 1000)))
    out_path = os.environ.get(
        'OUT', os.path.join(model_dir, 'final_eval_%dep.json' % episodes))

    env = utils.make_env(cfg)
    cfg.agent.params.obs_dim = env.observation_space.shape[0]
    cfg.agent.params.action_dim = env.action_space.shape[0]
    cfg.agent.params.action_range = [
        float(env.action_space.low.min()),
        float(env.action_space.high.max()),
    ]
    agent = hydra.utils.instantiate(cfg.agent)
    agent.load(model_dir, step)          # actor_<step>.pt, critic_*, critic_target_*

    returns, lengths = [], []
    for ep in range(episodes):
        env.seed(seed_base + ep)         # fixed episodes -> reruns reproduce
        obs = env.reset()
        agent.reset()
        done = False
        ret, n = 0.0, 0
        while not done:
            with utils.eval_mode(agent):
                action = agent.act(obs, sample=False)
            obs, reward, done, _ = env.step(action)
            ret += float(reward)
            n += 1
        returns.append(ret)
        lengths.append(n)
    env.close()

    r = np.asarray(returns)
    summary = dict(
        model_dir=model_dir,
        step=str(step),
        episodes=episodes,
        deterministic=True,
        eval_seed_base=seed_base,
        env=str(cfg.env),
        max_episode_steps=int(cfg.max_episode_steps) if cfg.max_episode_steps else None,
        train_seed=int(cfg.seed),
        reward_source='ground_truth_env_reward',
        mean_return=float(r.mean()),
        sd_return=float(r.std(ddof=1)),
        se_return=float(r.std(ddof=1) / np.sqrt(len(r))),
        median_return=float(np.median(r)),
        frac_return_above_200=float((r > 200).mean()),
        mean_episode_length=float(np.mean(lengths)),
        ep_returns=returns,
    )
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != 'ep_returns'}, indent=2))
    print('wrote %s' % out_path)


if __name__ == '__main__':
    main()
