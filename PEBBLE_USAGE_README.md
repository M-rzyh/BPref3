# PEBBLE on LunarLander — Usage Guide

## Prerequisites

```bash
conda activate bpref39_clone
```

Every run requires the `COMPARE_RUN_DIR` environment variable:

```bash
export COMPARE_RUN_DIR="$SCRATCH/compare_runs/pebble/lunarlander/<run_name>"
mkdir -p "$COMPARE_RUN_DIR"
```

---

## 1. Train with Oracle (Synthetic) Labels

The simplest mode — an automated teacher labels preferences:

```bash
bash scripts/lunar_lander/500/oracle/run_PEBBLE.sh
```

This runs `train_PEBBLE.py` with `feed_type=0` (oracle). Key parameters in the script:

| Parameter | Value | Meaning |
|---|---|---|
| `num_unsup_steps` | 9000 | Unsupervised exploration before reward learning |
| `num_train_steps` | 1000000 | Total training budget |
| `num_interact` | 20000 | Steps between preference queries |
| `max_feedback` | 1000 | Max preference labels to collect |
| `reward_batch` | 25 | Preference pairs per query batch |
| `segment` | 50 | Length of trajectory segments being compared (set in config) |

Pass a different feed type as an argument:

```bash
bash scripts/lunar_lander/500/oracle/run_PEBBLE.sh 2   # entropy sampling
```

Feed types: `0`=uniform, `1`=disagreement, `2`=entropy, `3`=kcenter, `4`=kcenter+disagree, `5`=kcenter+entropy.

---

## 2. Train with Inline Human Labels (feed_type=6)

Training pauses at each query batch and asks you to label in the terminal:

```bash
bash scripts/lunar_lander/500/oracle/run_PEBBLE.sh 6
```

At each query point you see pairs of trajectory segments described as text (position, velocity, angle, leg contact) and choose:

- `1` — Segment A is better
- `2` — Segment B is better
- `0` — Equal
- `s` — Skip
- `q` — Quit and save what you have

Note: This is text-only (numerical obs/action summaries). It works but can be hard to judge without visuals.

---

## 3. Offline Human Labeling with Videos (Recommended)

### Step 1: Run training with query video saving

```bash
export SAVE_QUERY_VIDEOS=true
bash scripts/lunar_lander/500/oracle/run_PEBBLE.sh 0
```

This saves `.mp4` videos of each segment pair under `$COMPARE_RUN_DIR/query_videos/batch_NNN/pair_NNN/`.

### Step 2: Label them offline

```bash
python human_label.py \
    --query_dir $COMPARE_RUN_DIR/query_videos \
    --output human_labels.pkl
```

Plays each pair of videos (Segment A, Segment B) and you label them. Same keys as inline mode, plus `r` to replay videos.

Outputs:
- `human_labels.pkl` — labels loadable by `reward_model.load_human_labels()`
- `human_labels.csv` — timing log (per-label duration, cumulative time)

### Step 3 (optional): Load labels into a reward model

```python
reward_model.load_human_labels('human_labels.pkl')
```

---

## 4. Offline Text-Based Labeling (from Saved Trajectories)

After any training run, trajectories are auto-saved to `<work_dir>/trajectories.pkl`. You can label them after the fact:

```bash
python human_label.py \
    --traj_file /path/to/trajectories.pkl \
    --output human_labels.pkl \
    --num_queries 50 \
    --segment 50
```

---

## 5. Collect Human Demos for GAIL (human_demo.py)

Separate from PEBBLE — this records expert demonstrations for GAIL.

### Curses mode (real-time keyboard, works over SSH)

```bash
python human_demo.py \
    --env LunarLanderContinuous-v2 \
    --mode curses \
    --max_episodes 10 \
    --fps 10
```

Controls:
- `W` / Up — main engine
- `A` / Left — left thrust
- `D` / Right — right thrust
- `S` / Down — coast
- `SPACE` — end episode (save it)
- `X` — discard episode
- `Q` — save all and quit

### Text mode (step-by-step, slower but precise)

```bash
python human_demo.py --env LunarLanderContinuous-v2 --mode text
```

### Append more episodes to an existing file

```bash
python human_demo.py \
    --env LunarLanderContinuous-v2 \
    --append demos/LunarLanderContinuous-v2/demos.pkl
```

### Convert to IRL3/imitation format for GAIL

```bash
python human_demo.py \
    --convert demos/LunarLanderContinuous-v2/demos.pkl \
    --export_imitation demos/LunarLanderContinuous-v2/imitation/
```

---

## 6. Monitoring Training

Logs go to:
- `$COMPARE_RUN_DIR/pebble/` — PEBBLE-specific logs
- `$COMPARE_RUN_DIR/common_tb/` — shared TensorBoard scalars
- `$COMPARE_RUN_DIR/pebble/videos/` — policy videos (if `save_video=true`)

View with:

```bash
tensorboard --logdir $COMPARE_RUN_DIR
```

---

## 7. Other Teacher Types

Pre-made scripts for different simulated teacher behaviors:

```
scripts/lunar_lander/500/mistake/run_PEBBLE.sh   # teacher makes labeling mistakes
scripts/lunar_lander/500/noisy/run_PEBBLE.sh     # noisy teacher
scripts/lunar_lander/500/skip/run_PEBBLE.sh      # teacher skips some queries
scripts/lunar_lander/500/equal/run_PEBBLE.sh     # teacher says "equal" sometimes
scripts/lunar_lander/500/myopic/run_PEBBLE.sh    # short-sighted teacher
```

1000-feedback variants are under `scripts/lunar_lander/1000/`.
