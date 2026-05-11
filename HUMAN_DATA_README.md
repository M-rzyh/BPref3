# Human Data Collection Tools

Terminal-based tools for collecting human preferences (PEBBLE) and demonstrations (GAIL), alongside the existing oracle system.

---

## 1. Preference Labeling for PEBBLE (`human_label.py`)

Shows two trajectory segments side-by-side with semantic summaries (position, velocity, angle, legs for LunarLander; raw obs for other envs).

Input keys: `[1]` A better, `[2]` B better, `[0]` equal, `[s]` skip, `[q]` quit & save

### Inline mode (during training)

Set `feed_type=6` in your run script. Training pauses at each query batch and prompts for your input in the terminal.

```bash
python train_PEBBLE.py \
  env=gym_LunarLanderContinuous-v2 \
  feed_type=6 \
  reward_batch=10 \
  ...
```

(Use a small `reward_batch` like 10 so you don't have to label too many at once.)

### Offline mode (label saved trajectories after training)

Every PEBBLE run now saves `trajectories.pkl` in the work directory at the end of training. You can label those offline:

```bash
python human_label.py \
  --traj_file <run_dir>/trajectories.pkl \
  --output human_labels.pkl \
  --num_queries 50 \
  --segment 50
```

Then load the labels into a new or resumed run via Python:

```python
reward_model.load_human_labels("human_labels.pkl")
```

---

## 2. Demonstration Collection for GAIL (`human_demo.py`)

Records human-controlled episodes for use as expert demonstrations in GAIL.

### Curses mode (real-time keyboard, default)

```bash
python human_demo.py \
  --env LunarLanderContinuous-v2 \
  --output demos/LunarLander/demos.pkl \
  --max_episodes 20 \
  --fps 10 \
  --mode curses
```

Controls:
- `W` / Up arrow : main engine ON
- `A` / Left arrow : thrust left
- `D` / Right arrow : thrust right
- `S` / Down arrow : coast (no action)
- `SPACE` : end current episode and save it
- `X` : discard current episode
- `Q` : save all collected demos and quit

### Text mode (step-by-step, for when curses doesn't work)

```bash
python human_demo.py --env LunarLanderContinuous-v2 --mode text
```

At each step you can type a shortcut key (`w`, `a`, `d`, `s`) or exact values like `0.5,-1.0`.

### Appending to existing demos

```bash
python human_demo.py \
  --env LunarLanderContinuous-v2 \
  --output demos/LunarLander/demos.pkl \
  --append demos/LunarLander/demos.pkl
```

### Demo file format

The output `.pkl` file contains a Python list of episode dicts:

```python
[
  {
    'observations':      np.array (T, obs_dim),
    'next_observations': np.array (T, obs_dim),
    'actions':           np.array (T, act_dim),
    'rewards':           np.array (T,),
    'dones':             np.array (T,),
  },
  ...
]
```

---

## What changed in existing files

All changes are **additions only** -- no existing behavior was modified or removed.

| File | Change |
|---|---|
| `reward_model.py` | Added 3 methods at the bottom: `human_sampling()`, `save_trajectories()`, `load_human_labels()` |
| `train_PEBBLE.py` | Added `feed_type == 6` case in `learn_reward()` (1 line). Added `save_trajectories()` call at end of training (1 line). |

The existing oracle labeling flow (`feed_type` 0--5) is completely untouched.
