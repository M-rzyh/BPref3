# Unified-Interactive PEBBLE Web Run + Split Comparison

End-to-end recipe: collect 100 human labels via web AND finish pure-RL in **one
interactive job**, while a **separate sbatch resume job** trains its own copy
from the same checkpoint. Lets you compare unified vs split workflows.

The `run_cc_lunar_smoke.sh` smoke validates the underlying save-and-continue
plumbing in ~50 sec, run it first if you haven't.

## 1. Compute node — start training (interactive, 8 h)

```bash
salloc --account=aip-mtaylor3 --gres=gpu:1 --cpus-per-task=8 --mem=16G --time=08:00:00
```

Then on the compute node:

```bash
cd ~/BPref3
module --force purge && module load StdEnv/2023
eval "$(/scratch/marzii/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch/marzii/envs/bpref39_clone
./scripts/lunar_lander/500/web_full/run_PEBBLE.sh 2>&1 | tee ~/BPref3/logs/lunarlander/web_full_interactive.out
```

Note the JobID printed (`COMPARE_RUN_DIR=...lunarlander_web_full/<JOBID>/seed_12345`).

## 2. Login node — start the web UI

In a separate VS Code terminal on the login node:

```bash
PARENT=/scratch/marzii/compare_runs/pebble/lunarlander_web_full/<JOBID>

OPENBLAS_NUM_THREADS=1 python /home/marzii/BPref3/label_web.py --mode online \
  --online_query_dir "$PARENT" \
  --output "$PARENT/human_labels.pkl" \
  --port 8888
```

VS Code → **Ports** tab → forward `8888` → open `http://localhost:8888/online`.

## 3. Label

| Key | Action |
|-----|--------|
| SPACE | Play |
| 1 | A better |
| 2 | B better |
| 0 | Equal |
| s | Skip |
| r | Replay |

After the 4 batches × 25 labels = **100 labels** are collected, the training
log will show:

```
[checkpoint] max_feedback hit (total=100). Saving full training state...
[checkpoint] saved -> .../checkpoint.pt
[checkpoint] exit_after_max_feedback_checkpoint=false -> continuing training to num_train_steps
```

The web UI can be closed. Training continues in the same interactive session
all the way to `num_train_steps=1,000,000`.

## 4. After max_feedback hits — submit the split-workflow comparison job

In a third terminal (login node), as soon as the `[checkpoint] saved` line
appears, copy the checkpoint to a sibling dir and submit the resume sbatch:

```bash
UJ=<UNIFIED_JOBID>
SRC=/scratch/marzii/compare_runs/pebble/lunarlander_web_full/$UJ/seed_12345/pebble
COMPARE=/scratch/marzii/compare_runs/pebble/lunarlander_web_full_compare/$UJ/seed_12345/pebble
mkdir -p "$COMPARE"
cp -p "$SRC/checkpoint.pt"* "$COMPARE/"
RESUME_FROM="$COMPARE/checkpoint.pt" sbatch run_cc_lunar_resume.sh
```

The copy is required so the resume job's logs/CSVs/TB don't collide with the
unified run's still-active work_dir.

## 5. When both finish — compare

Two `train.csv` files:

| Run | Path |
|---|---|
| Unified (interactive, 0 → 1M in one job) | `lunarlander_web_full/$UJ/seed_12345/pebble/train.csv` |
| Split-comparison (resume, ~70K → 1M as sbatch) | `lunarlander_web_full_compare/$UJ/seed_12345/pebble/train.csv` |

Plot side-by-side with the existing multi-seed plotter:

```bash
python /home/marzii/BPref3/scripts/plot_pebble_multi_seed.py \
  --group "Unified (interactive):$UJ" \
  --group "Split (resume):$UJ" \
  --root /scratch/marzii/compare_runs/pebble/lunarlander_web_full \
  --output /scratch/marzii/compare_runs/pebble/unified_vs_split_$UJ.png \
  --ylim=-600,400
```

(Adjust `--root` per group if needed — the plotter currently takes one root,
so for an apples-to-apples curve you may need to call it once per location and
overlay the PNGs, or copy both train.csv files into a common root.)

## What's saved at each checkpoint

**Mid-run (when max_feedback is hit):**
- `checkpoint.pt` (SAC + reward model + optimizers + RNG + counters)
- `checkpoint.pt.replay.npz` (replay buffer)
- `checkpoint.pt.meta.json` (manifest)

**End of run (after num_train_steps):**
- `actor_<step>.pt`, `critic_<step>.pt`, `critic_target_<step>.pt`
- `reward_model_<step>_*.pt`
- `trajectories.pkl`
- `videos/final_ep0.mp4` … `final_ep4.mp4`
- `compute_time.json` (full timing breakdown)
- `train.csv`, `eval.csv`, `tb/` events (rolling)

## Notes

- Web mode renders one `.mp4` per segment for the browser. Xvfb is started
  automatically because `feed_type=7`.
- The unified run uses `checkpoint_at_max_feedback=true` and
  `exit_after_max_feedback_checkpoint=false` (the new flag) so the same job
  saves the checkpoint AND keeps training.
- `cumulative_wait_for_start_sec` records time the web batch was open before
  you pressed Play. `human_pref_sec` records actual decision time. Both land
  in `compute_time.json` plus per-batch in the printed table.

## 6. Watching both runs live in TensorBoard

On the login node (substitute `$UJ` with the unified job ID):

```bash
tensorboard --logdir_spec "unified:/scratch/marzii/compare_runs/pebble/lunarlander_web_full/$UJ/seed_12345/pebble/tb,compare:/scratch/marzii/compare_runs/pebble/lunarlander_web_full_compare/$UJ/seed_12345/pebble/tb" --port 6010 --host 127.0.0.1
```

VS Code → **Ports** tab → forward `6010` → open `http://localhost:6010`.

You'll see two named runs (`unified`, `compare`) overlaid on every chart:
`train/episode_reward`, `train/true_episode_reward`, `train/critic_loss`,
`train/total_feedback`, etc. Auto-refreshes every ~30s.

Notes:
- Don't use `--bind_all` — there's a TensorBoard bug
  (`UnboundLocalError: local variable 'host' referenced before assignment`)
  on this version that triggers on `--bind_all`.
- If port 6010 is busy, pick another (6011, 7654, …) and update the VS Code
  forward.
