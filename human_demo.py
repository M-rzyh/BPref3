#!/usr/bin/env python3
"""
Terminal-based demonstration collection for GAIL.

Provides keyboard control of a gym environment so a human can record
expert demonstrations.  Works over SSH (curses mode) — no display needed.

Supports both continuous (LunarLanderContinuous-v2) and discrete (LunarLander-v2)
action spaces.

Usage:
    python human_demo.py --env LunarLanderContinuous-v2 \
        [--output demos/LunarLander/demos.pkl] \
        [--max_episodes 10] [--fps 10] [--mode curses]

    python human_demo.py --env LunarLander-v2 \
        [--output demos/LunarLander/demos_discrete.pkl]

Controls (curses mode):
    W / Up    : main engine ON  (continuous: [1,0], discrete: action 2)
    A / Left  : thrust left     (continuous: [0,-1], discrete: action 1)
    D / Right : thrust right    (continuous: [0,1], discrete: action 3)
    S / Down  : no-op / coast   (continuous: [0,0], discrete: action 0)
    SPACE     : end current episode (save it)
    X         : discard current episode
    Q         : save all & quit

Controls (text mode):
    Enter an action each step as a shortcut key (w/a/d/s) or:
      - Continuous: "main,lateral" e.g. "1.0,-0.5"
      - Discrete: action index e.g. "2"
"""

import os
import sys
import time
import argparse
import pickle
import numpy as np
from datetime import datetime

try:
    import gym
except ImportError:
    sys.exit("gym is required: pip install gym")


# ---------------------------------------------------------------------------
# Action mapping for LunarLanderContinuous-v2
# ---------------------------------------------------------------------------
PRESETS_CONTINUOUS = {
    'w':     np.array([ 1.0,  0.0], dtype=np.float32),   # main engine
    'a':     np.array([ 0.0, -1.0], dtype=np.float32),   # left
    'd':     np.array([ 0.0,  1.0], dtype=np.float32),   # right
    's':     np.array([ 0.0,  0.0], dtype=np.float32),   # coast
    'wa':    np.array([ 1.0, -1.0], dtype=np.float32),   # main + left
    'wd':    np.array([ 1.0,  1.0], dtype=np.float32),   # main + right
    'none':  np.array([ 0.0,  0.0], dtype=np.float32),   # coast
}

# ---------------------------------------------------------------------------
# Action mapping for LunarLander-v2 (discrete)
# 0=noop, 1=fire left, 2=fire main, 3=fire right
# ---------------------------------------------------------------------------
PRESETS_DISCRETE = {
    'w': 2,   # main engine
    'a': 1,   # left engine
    'd': 3,   # right engine
    's': 0,   # noop
    'none': 0,
}


def _state_string(obs, step, ep_reward, act=None):
    """One-line (or multi-line) state summary for LunarLander (obs_dim=8)."""
    lines = []
    if len(obs) == 8:
        lines.append(f"  Step {step:4d}  |  Ep reward: {ep_reward:+8.2f}")
        lines.append(f"  Pos  x={obs[0]:+.3f}  y={obs[1]:+.3f}")
        lines.append(f"  Vel  vx={obs[2]:+.3f}  vy={obs[3]:+.3f}")
        lines.append(f"  Angle {obs[4]:+.3f} rad ({np.degrees(obs[4]):+.1f} deg)   angvel={obs[5]:+.3f}")
        l, r = 'Y' if obs[6] > 0.5 else 'N', 'Y' if obs[7] > 0.5 else 'N'
        lines.append(f"  Legs  L={l}  R={r}")
        if act is not None:
            if isinstance(act, (int, np.integer)):
                names = {0: 'noop', 1: 'left', 2: 'main', 3: 'right'}
                lines.append(f"  Last action: {act} ({names.get(int(act), '?')})")
            else:
                lines.append(f"  Last action: main={act[0]:+.2f}  lat={act[1]:+.2f}")
    else:
        n = min(8, len(obs))
        vals = ' '.join(f'{v:+.3f}' for v in obs[:n])
        lines.append(f"  Step {step:4d}  |  Ep reward: {ep_reward:+.2f}")
        lines.append(f"  Obs[:{n}]: {vals}{'...' if len(obs)>n else ''}")
        if act is not None:
            lines.append(f"  Action: {act}")
    return '\n'.join(lines)


def _is_discrete(env):
    """Check if action space is discrete."""
    return isinstance(env.action_space, gym.spaces.Discrete)


# ---------------------------------------------------------------------------
# Curses-based real-time demo collection
# ---------------------------------------------------------------------------

def _run_curses(env, max_episodes, fps, discrete):
    import curses

    demos = []
    ep_times = []

    def _inner(stdscr):
        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.timeout(int(1000 / fps))

        ep = 0
        while ep < max_episodes:
            ep_start = time.time()
            obs = env.reset()
            ep_obs, ep_act, ep_rew, ep_done = [obs.copy()], [], [], []
            step = 0
            ep_reward = 0.0
            done = False
            last_act = 0 if discrete else np.zeros(env.action_space.shape[0], dtype=np.float32)
            discard = False

            while not done:
                stdscr.clear()
                stdscr.addstr(0, 0, f"{'=' * 56}")
                mode_str = "DISCRETE" if discrete else "CONTINUOUS"
                stdscr.addstr(1, 0, f"  DEMO [{mode_str}]  |  Ep {ep+1}/{max_episodes}  |  Saved: {len(demos)}")
                stdscr.addstr(2, 0, f"{'=' * 56}")
                for i, line in enumerate(_state_string(obs, step, ep_reward, last_act).split('\n')):
                    stdscr.addstr(4 + i, 0, line)
                row = 4 + i + 2
                stdscr.addstr(row, 0, "  W=up  A=left  D=right  S=coast")
                stdscr.addstr(row+1, 0, "  SPACE=end episode  X=discard  Q=quit")
                stdscr.refresh()

                key = stdscr.getch()

                quit_all = False

                if key == ord('q') or key == ord('Q'):
                    quit_all = True
                    break
                elif key == ord('x') or key == ord('X'):
                    discard = True
                    break
                elif key == ord(' '):
                    done = True
                    break

                # Map key to action
                if discrete:
                    if key == ord('w') or key == curses.KEY_UP:
                        action = 2  # main engine
                    elif key == ord('a') or key == curses.KEY_LEFT:
                        action = 1  # left
                    elif key == ord('d') or key == curses.KEY_RIGHT:
                        action = 3  # right
                    else:
                        action = 0  # noop
                else:
                    act_dim = env.action_space.shape[0]
                    action = np.zeros(act_dim, dtype=np.float32)
                    if key == ord('w') or key == curses.KEY_UP:
                        action[0] = 1.0
                    elif key == ord('a') or key == curses.KEY_LEFT:
                        if act_dim > 1:
                            action[1] = -1.0
                    elif key == ord('d') or key == curses.KEY_RIGHT:
                        if act_dim > 1:
                            action[1] = 1.0
                    elif key == ord('s') or key == curses.KEY_DOWN:
                        pass  # coast

                next_obs, reward, env_done, info = env.step(action)
                last_act = action if discrete else action.copy()
                ep_act.append(action if discrete else action.copy())
                ep_rew.append(reward)
                ep_done.append(env_done)

                obs = next_obs
                ep_obs.append(obs.copy())
                step += 1
                ep_reward += reward

                if env_done:
                    done = True

            ep_end = time.time()

            if quit_all:
                if len(ep_act) > 0 and not discard:
                    demos.append(_pack_episode(ep_obs, ep_act, ep_rew, ep_done, discrete))
                    ep_times.append(ep_end - ep_start)
                break

            if discard:
                stdscr.clear()
                stdscr.addstr(0, 0, f"  Episode {ep+1} discarded.")
                stdscr.refresh()
                curses.napms(600)
                continue

            if len(ep_act) > 0:
                demos.append(_pack_episode(ep_obs, ep_act, ep_rew, ep_done, discrete))
                ep_times.append(ep_end - ep_start)

            stdscr.clear()
            stdscr.addstr(0, 0, f"  Episode {ep+1} done: {step} steps, reward={ep_reward:+.2f}")
            stdscr.addstr(1, 0, f"  Total saved: {len(demos)} episodes")
            stdscr.refresh()
            curses.napms(1200)
            ep += 1

    curses.wrapper(_inner)
    return demos, ep_times


# ---------------------------------------------------------------------------
# Text-based fallback
# ---------------------------------------------------------------------------

def _run_text(env, max_episodes, discrete):
    demos = []
    ep_times = []

    for ep in range(max_episodes):
        ep_start = time.time()
        obs = env.reset()
        ep_obs, ep_act, ep_rew, ep_done = [obs.copy()], [], [], []
        step = 0
        ep_reward = 0.0
        done = False
        discard = False

        print(f"\n{'=' * 56}")
        print(f"  Episode {ep+1}/{max_episodes}  |  Saved so far: {len(demos)}")
        act_type = "DISCRETE (0=noop, 1=left, 2=main, 3=right)" if discrete else "CONTINUOUS"
        print(f"  Action space: {act_type}")
        print(f"{'=' * 56}")

        while not done:
            print(_state_string(obs, step, ep_reward))
            if discrete:
                print("  Enter: w/a/d/s or action index (0-3), space=end, x=discard, q=quit")
            else:
                print("  Enter: w/a/d/s, or 'main,lat' (e.g. 0.5,-1), space=end, x=discard, q=quit")

            try:
                raw = input("  > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                raw = 'q'

            if raw == 'q':
                if len(ep_act) > 0 and not discard:
                    demos.append(_pack_episode(ep_obs, ep_act, ep_rew, ep_done, discrete))
                    ep_times.append(time.time() - ep_start)
                return demos, ep_times
            elif raw == 'x':
                discard = True
                print("  Discarded.")
                break
            elif raw in ('', 'space', ' '):
                break

            if discrete:
                if raw in PRESETS_DISCRETE:
                    action = PRESETS_DISCRETE[raw]
                else:
                    try:
                        action = int(raw)
                        if action < 0 or action >= env.action_space.n:
                            print(f"  (out of range, using noop)")
                            action = 0
                    except ValueError:
                        print("  (unrecognized, using noop)")
                        action = 0
            else:
                act_dim = env.action_space.shape[0]
                if raw in PRESETS_CONTINUOUS:
                    action = PRESETS_CONTINUOUS[raw].copy()
                    if act_dim != 2:
                        action = np.zeros(act_dim, dtype=np.float32)
                        action[0] = PRESETS_CONTINUOUS[raw][0]
                else:
                    try:
                        vals = [float(x) for x in raw.replace(' ', ',').split(',')]
                        action = np.zeros(act_dim, dtype=np.float32)
                        for j, v in enumerate(vals[:act_dim]):
                            action[j] = np.clip(v, -1.0, 1.0)
                    except ValueError:
                        print("  (unrecognized, using coast)")
                        action = np.zeros(act_dim, dtype=np.float32)

            next_obs, reward, env_done, info = env.step(action)
            ep_act.append(action if discrete else action.copy())
            ep_rew.append(reward)
            ep_done.append(env_done)
            obs = next_obs
            ep_obs.append(obs.copy())
            step += 1
            ep_reward += reward

            if env_done:
                done = True

        ep_end = time.time()

        if discard:
            continue

        if len(ep_act) > 0:
            demos.append(_pack_episode(ep_obs, ep_act, ep_rew, ep_done, discrete))
            ep_times.append(ep_end - ep_start)
            print(f"  Saved episode {ep+1}: {step} steps, reward={ep_reward:+.2f}")

    return demos, ep_times


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pack_episode(obs_list, act_list, rew_list, done_list, discrete=False):
    """Pack one episode into a dict of numpy arrays."""
    if discrete:
        actions = np.array(act_list, dtype=np.int64)
    else:
        actions = np.array(act_list, dtype=np.float32)

    return {
        'observations': np.array(obs_list[:-1], dtype=np.float32),
        'next_observations': np.array(obs_list[1:], dtype=np.float32),
        'actions': actions,
        'rewards': np.array(rew_list, dtype=np.float32),
        'dones': np.array(done_list, dtype=np.float32),
    }


def save_demos(demos, path, timing=None):
    """Save demos with optional timing metadata.

    New format: {'demos': [...], 'timing': {...}}
    Old format (if timing is None): just the list of episode dicts
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

    if timing is not None:
        data = {'demos': demos, 'timing': timing}
    else:
        data = demos

    with open(path, 'wb') as f:
        pickle.dump(data, f)

    total_steps = sum(len(d['rewards']) for d in demos)
    total_reward = sum(d['rewards'].sum() for d in demos)
    print(f"\nSaved {len(demos)} episodes ({total_steps} steps, total reward={total_reward:+.1f}) -> {path}")
    if timing:
        print(f"Total demo collection time: {timing['total_time_sec']:.1f} sec")


def load_demos(path):
    """Load demos, auto-detecting old (list) or new (dict with timing) format."""
    with open(path, 'rb') as f:
        data = pickle.load(f)

    if isinstance(data, list):
        # Old format: just a list of episode dicts
        return data, None
    elif isinstance(data, dict) and 'demos' in data:
        # New format with timing
        return data['demos'], data.get('timing')
    else:
        raise ValueError(f"Unrecognized demo format in {path}")


def save_timing_csv(csv_path, ep_times, demos):
    """Save per-episode timing CSV log."""
    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
    cumulative = 0.0
    with open(csv_path, 'w') as f:
        f.write("episode_idx,duration_sec,cumulative_sec,ep_reward,ep_length\n")
        for i, (dt, demo) in enumerate(zip(ep_times, demos)):
            cumulative += dt
            ep_reward = float(demo['rewards'].sum())
            ep_length = len(demo['rewards'])
            f.write(f"{i},{dt:.3f},{cumulative:.3f},{ep_reward:.2f},{ep_length}\n")
    print(f"Saved timing CSV -> {csv_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Collect human demonstrations for GAIL")
    parser.add_argument('--env', type=str, default='LunarLanderContinuous-v2',
                        help='Gym environment ID (default: LunarLanderContinuous-v2)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output .pkl path (default: demos/<env>/demos.pkl)')
    parser.add_argument('--max_episodes', type=int, default=20,
                        help='Max episodes to collect')
    parser.add_argument('--fps', type=int, default=10,
                        help='Target FPS for curses mode (default: 10)')
    parser.add_argument('--mode', type=str, choices=['curses', 'text'], default='curses',
                        help='Input mode: curses (real-time keyboard) or text (step-by-step)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--append', type=str, default=None,
                        help='Path to existing demos.pkl to append to')
    parser.add_argument('--csv', type=str, default=None,
                        help='Path for timing CSV log (default: <output>.csv)')

    # Format converter
    parser.add_argument('--convert', type=str, default=None,
                        help='Path to existing demos.pkl to convert (no collection)')
    parser.add_argument('--export_imitation', type=str, default=None,
                        help='Export demos in IRL3/imitation Trajectory format to this dir')

    args = parser.parse_args()

    # --- Format conversion mode ---
    if args.convert and args.export_imitation:
        _convert_to_imitation(args.convert, args.export_imitation)
        return

    # --- Demo collection mode ---
    env = gym.make(args.env)
    env.seed(args.seed)
    discrete = _is_discrete(env)

    if discrete:
        print(f"Environment: {args.env}  (obs={env.observation_space.shape[0]}, discrete actions: {env.action_space.n})")
    else:
        print(f"Environment: {args.env}  (obs={env.observation_space.shape[0]}, act={env.action_space.shape[0]})")
    print(f"Mode: {args.mode}   Max episodes: {args.max_episodes}")

    session_start = time.time()

    if args.mode == 'curses':
        demos, ep_times = _run_curses(env, args.max_episodes, args.fps, discrete)
    else:
        demos, ep_times = _run_text(env, args.max_episodes, discrete)

    session_end = time.time()
    env.close()

    if args.append and os.path.exists(args.append):
        old_demos, _ = load_demos(args.append)
        print(f"Appending to {args.append} ({len(old_demos)} existing episodes)")
        demos = old_demos + demos

    if len(demos) == 0:
        print("No demos collected.")
        return

    # Build timing metadata
    timing = {
        'total_time_sec': session_end - session_start,
        'per_episode_times_sec': ep_times,
        'session_start': datetime.fromtimestamp(session_start).isoformat(),
    }

    out_path = args.output or f"demos/{args.env}/demos.pkl"
    save_demos(demos, out_path, timing=timing)

    # Save CSV log
    csv_path = args.csv or (out_path.replace('.pkl', '') + '.csv')
    save_timing_csv(csv_path, ep_times, demos[-len(ep_times):])


# ---------------------------------------------------------------------------
# Format converter: BPref3 demos -> IRL3/imitation Trajectory format
# ---------------------------------------------------------------------------

def _convert_to_imitation(demos_path, output_dir):
    """Convert BPref3 demo pkl to IRL3 imitation Trajectory pickle format.

    IRL3 Trajectory format:
        obs: (T+1, obs_dim)  — includes initial obs
        acts: (T, act_dim)
        infos: None
        terminal: True
    """
    demos, timing = load_demos(demos_path)
    print(f"Loaded {len(demos)} episodes from {demos_path}")

    trajectories = []
    for i, demo in enumerate(demos):
        obs = demo['observations']
        next_obs = demo['next_observations']
        acts = demo['actions']

        # Validate
        if len(obs) == 0:
            print(f"  [warn] episode {i} is empty, skipping")
            continue
        if obs.shape[0] != acts.shape[0]:
            print(f"  [warn] episode {i} shape mismatch obs={obs.shape} acts={acts.shape}, skipping")
            continue
        if np.isnan(obs).any() or np.isnan(acts).any():
            print(f"  [warn] episode {i} contains NaN, skipping")
            continue

        # Build Trajectory-compatible dict:
        # obs = (T+1, obs_dim): stack observations + last next_obs
        traj_obs = np.vstack([obs, next_obs[-1:]])
        traj = {
            'obs': traj_obs,
            'acts': acts,
            'infos': None,
            'terminal': True,
        }
        trajectories.append(traj)

    if len(trajectories) == 0:
        print("No valid trajectories to convert.")
        return

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'demos.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(trajectories, f)

    # Also save timing if available
    if timing:
        timing_path = os.path.join(output_dir, 'timing.pkl')
        with open(timing_path, 'wb') as f:
            pickle.dump(timing, f)
        print(f"Saved timing -> {timing_path}")

    print(f"Converted {len(trajectories)} trajectories -> {out_path}")


if __name__ == '__main__':
    main()
