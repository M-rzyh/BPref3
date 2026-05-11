# PEBBLE Checkpoint & Crash Recovery

## What gets saved

When `checkpoint_at_max_feedback=true`, the run auto-saves and exits the moment `total_feedback >= max_feedback`. Two files in `<work_dir>/`:

- `checkpoint.pt` — SAC weights, reward model weights, all optimizer states, preference buffer, trajectory buffer, RNG state, all counters
- `checkpoint.pt.replay.npz` — replay buffer arrays (numpy, ~600 MB, no pickle)
- `checkpoint.pt.meta.json` — sidecar listing what's inside

Both written via `tmp + rename`, so a kill mid-write never leaves a partial file.

## Resuming pure RL from a checkpoint

```bash
RESUME_FROM=/scratch/marzii/compare_runs/pebble/lunarlander_web/<JOB>/seed_<SEED>/pebble/checkpoint.pt \
  sbatch run_cc_lunar_resume.sh
```

Continues the same `train.csv`, `eval.csv`, and TensorBoard `tb/` directory (append mode), so plots show one continuous run.

## Crash recovery (replay saved labels)

If a run died **after** you finished labeling but **before** the checkpoint succeeded (e.g. OOM during `torch.save`), the labels are still on disk in `online_queries/batch_*/response.pkl`. Recover with:

```bash
REPLAY_DIR=/scratch/marzii/compare_runs/pebble/lunarlander_web/<OLD_JOB>/seed_<SEED>/online_queries \
  sbatch run_cc_lunar_replay.sh
```

The replay job:
- Reads each saved batch's `metadata.pkl` + `response.pkl` directly into the preference buffer (no rendering, no human waiting)
- Falls through to live `web_sampling` for any batch number not on disk
- Exits with a fresh checkpoint when `max_feedback` is hit (~15-20 min on GPU)

## Memory note

Always request **at least 16 GB**:
```bash
salloc --mem=16G ...
# or in sbatch
#SBATCH --mem=16G
```
The replay buffer alone is ~600 MB and `torch.save` momentarily doubles the active footprint.
