# Interactive PEBBLE Web Labeling — Unified (full) mode

This recipe uses the **unified workflow**: one job collects 100 human labels via
the web UI AND continues training to `num_train_steps=1,000,000` in the same
session, saving a `checkpoint.pt` at the max_feedback boundary so a separate
sbatch can later replay just the pure-RL portion for comparison.

> If you want the **split workflow** instead (job exits at max_feedback,
> resume in a separate sbatch), swap the launch script in step 1 from
> `web_full/run_PEBBLE.sh` to `web/run_PEBBLE.sh` — the rest of the recipe is
> identical.

## 1. Compute node — start training (full / unified mode)

```bash
salloc --account=aip-mtaylor3 --gres=gpu:1 --cpus-per-task=8 --mem=16G --time=06:00:00
```

(`--time=06:00:00` covers ~30 min labeling + ~4h45m pure RL with margin.)

Then on the compute node:

```bash
cd ~/BPref3
module --force purge
module load StdEnv/2023
eval "$(/scratch/marzii/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch/marzii/envs/bpref39_clone
hash -r

# UNIFIED workflow: write to lunarlander_web_full/<JOBID>/...
# (NOT lunarlander_web — that's the split-workflow path.)
export PARENT_RUN_DIR="$SCRATCH/compare_runs/pebble/lunarlander_web_full/$SLURM_JOB_ID"
mkdir -p "$PARENT_RUN_DIR"
LOG="$PARENT_RUN_DIR/run.out"
echo "PARENT_RUN_DIR=$PARENT_RUN_DIR"

./scripts/lunar_lander/500/web_full/run_PEBBLE.sh 2>&1 | tee "$LOG"
```

`web_full/run_PEBBLE.sh` sets `checkpoint_at_max_feedback=true` AND
`exit_after_max_feedback_checkpoint=false`, so the training writes a checkpoint
at the next episode boundary after max_feedback hits, then keeps training to
`num_train_steps`.

## 2. Login node (separate VS Code terminal) — start web UI

Look at the training output. It prints:

```
COMPARE_RUN_DIR=/scratch/marzii/compare_runs/pebble/lunarlander_web_full/<JOBID>/seed_<SEED>
```

Use the **parent dir** (drop `/seed_<SEED>`):

```bash
PARENT=/scratch/marzii/compare_runs/pebble/lunarlander_web_full/4895573

OPENBLAS_NUM_THREADS=1 python /home/marzii/BPref3/label_web.py --mode online \
  --online_query_dir "$PARENT" \
  --output "$PARENT/human_labels.pkl" \
  --port 7654
```

Replace `<JOBID>` with the actual SLURM_JOB_ID from the training output.
If port 7654 is busy, swap to another (e.g. 9999) and update VS Code port forward.

> ⚠️ **Path must match the training output**: training writes to
> `lunarlander_web_full/<JOBID>/seed_12345/online_queries/...` — your
> `--online_query_dir` must point at the parent (`lunarlander_web_full/<JOBID>`).
> If you use `lunarlander_web/<JOBID>` (without `_full`), label_web will see no
> batches and the training will sit idle forever waiting for labels.

## 3. Browser

VS Code → **Ports** tab → Forward port `7654` → open:

```
http://localhost:7654/online
```

Must include `/online`. Plain `http://localhost:7654/` returns 404.

## 4. Keys

| Key | Action |
|-----|--------|
| SPACE | Play |
| 1 | A better |
| 2 | B better |
| 0 | Equal |
| s | Skip |
| r | Replay |

## 5. After all labels collected → checkpoint saved + (optional) submit comparison sbatch

In **unified mode** (this recipe), the training keeps going past max_feedback. You'll see in the .out:

```
[checkpoint] max_feedback hit (total=100). Saving full training state...
[checkpoint] saved -> /scratch/marzii/.../lunarlander_web_full/<JOBID>/seed_12345/pebble/checkpoint.pt
[checkpoint] exit_after_max_feedback_checkpoint=false -> continuing training to num_train_steps
```

You can close the browser; the interactive job continues training to step 1M.

**Optional comparison sbatch** — to also run the split-workflow pure-RL phase from the same checkpoint (so you can verify unified vs split parity):

```bash
UJ=<UNIFIED_JOBID>
SRC=/scratch/marzii/compare_runs/pebble/lunarlander_web_full/$UJ/seed_12345/pebble
COMPARE=/scratch/marzii/compare_runs/pebble/lunarlander_web_full_compare/$UJ/seed_12345/pebble
mkdir -p "$COMPARE"
cp -p "$SRC/checkpoint.pt"* "$COMPARE/"
RESUME_FROM="$COMPARE/checkpoint.pt" sbatch --time=05:00:00 run_cc_lunar_resume.sh
```

The copy is required so the resume's logs/CSVs/TB don't collide with the still-running unified job's work_dir.

> If you ran the **split** workflow instead (`web/run_PEBBLE.sh`), the
> interactive job exits at max_feedback. Then submit the resume directly from
> the original checkpoint location:
> ```bash
> RESUME_FROM=/scratch/marzii/compare_runs/pebble/lunarlander_web/<JOBID>/seed_12345/pebble/checkpoint.pt \
>   sbatch run_cc_lunar_resume.sh
> ```

## Notes

- First query batch is always oracle (PEBBLE bootstrap). Web prompts begin at the second batch (~step 30K).
- Same web server handles all 5 seeds — point `--online_query_dir` at the parent dir, not a single seed.

## 6. Live TensorBoard (substitute your current `<UNIFIED_JOBID>`)

Once your fresh full run is running, view both runs (unified + optional split
comparison) overlaid:

```bash
UJ=<UNIFIED_JOBID>   # the SLURM_JOB_ID from your salloc

tensorboard --logdir_spec "unified:/scratch/marzii/compare_runs/pebble/lunarlander_web_full/$UJ/seed_12345/pebble/tb,compare:/scratch/marzii/compare_runs/pebble/lunarlander_web_full_compare/$UJ/seed_12345/pebble/tb" --port 6010 --host 127.0.0.1
```

(The `compare` directory will be empty until you submit the comparison sbatch
in step 5 — TB will silently show only the `unified` line until then.)

VS Code → **Ports** tab → forward `6010` → open `http://localhost:6010`.

Two named runs (`unified`, `compare`) overlay on every chart. Auto-refreshes
~every 30s. Key scalars: `train/episode_reward`, `train/true_episode_reward`,
`train/critic_loss`, `train/actor_loss`, `train/total_feedback`,
`train/labeled_feedback`.

Notes:
- Don't use `--bind_all` — TensorBoard version bug triggers
  `UnboundLocalError: local variable 'host' referenced before assignment`.
  `--host 127.0.0.1` is enough; VS Code port forwarding handles the rest.
- If port 6010 is busy, pick another (6011, 7654, …) and update the VS Code
  forward.
- The `compare` line starts at step ≈ where the unified run hit `max_feedback`
  and the checkpoint was taken. The `unified` line spans 0 → 1M.
