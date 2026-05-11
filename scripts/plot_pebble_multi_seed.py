#!/usr/bin/env python3
"""Plot mean±std PEBBLE learning curves across seeds, one band per group.

Reads `train.csv` (per-episode `step` + `true_episode_reward`) from each run,
smooths each seed with a rolling mean, interpolates onto a common step grid,
and plots mean ± std across seeds.

Usage:
    python plot_pebble_multi_seed.py \\
        --group "Oracle (max_feedback=100):4883370,4883371,4883372,4883373,4883374" \\
        --output pebble_5seed.png
"""

import argparse
import os
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_PEBBLE_ROOT = "/scratch/marzii/compare_runs/pebble/lunarlander"


def find_train_csv(job_id, root):
    """Auto-discover train.csv under <root>/<job_id>/. Supports two layouts:
      A) <root>/<JOBID>/pebble/train.csv               (oracle, one seed per job)
      B) <root>/<JOBID>/seed_<SEED>/pebble/train.csv   (web/full, may have many)
    Returns list of (seed_label, path).
    """
    job_dir = os.path.join(root, str(job_id))
    if not os.path.isdir(job_dir):
        return []
    found = []
    # Layout B: seed_* subdirs
    for seed_dir in sorted(glob.glob(os.path.join(job_dir, "seed_*"))):
        seed = os.path.basename(seed_dir).replace("seed_", "")
        csv_path = os.path.join(seed_dir, "pebble", "train.csv")
        if os.path.isfile(csv_path):
            found.append((seed, csv_path))
    # Layout A: pebble/ directly under job dir
    if not found:
        csv_path = os.path.join(job_dir, "pebble", "train.csv")
        if os.path.isfile(csv_path):
            found.append((str(job_id), csv_path))
    return found


def load_curve(csv_path, smooth_window=100):
    df = pd.read_csv(csv_path).sort_values("step")
    if "true_episode_reward" not in df.columns or "step" not in df.columns:
        raise ValueError(f"{csv_path}: missing required columns")
    steps = df["step"].to_numpy(dtype=float)
    reward = df["true_episode_reward"].to_numpy(dtype=float)
    if smooth_window > 1:
        reward = pd.Series(reward).rolling(smooth_window, min_periods=1).mean().to_numpy()
    return steps, reward


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--group", action="append", required=True,
                   help="Group spec 'Label:JOBID,JOBID,...' (repeatable)")
    p.add_argument("--output", required=True)
    p.add_argument("--title", default="PEBBLE learning curves — mean ± std across seeds")
    p.add_argument("--ylim", default=None, help="'min,max'")
    p.add_argument("--xlim", default=None, help="'min,max'")
    p.add_argument("--smooth", type=int, default=100,
                   help="Rolling-mean window over episodes (default 100)")
    p.add_argument("--grid", type=int, default=500,
                   help="Number of points on shared step grid for averaging (default 500)")
    p.add_argument("--root", default=DEFAULT_PEBBLE_ROOT,
                   help=f"Root dir holding <JOBID>/... subtrees (default {DEFAULT_PEBBLE_ROOT})")
    p.add_argument("--colors", default=None,
                   help="Comma-separated matplotlib colors, one per --group")
    p.add_argument("--show-seeds", action="store_true",
                   help="Plot each seed as a faint individual line")
    args = p.parse_args()

    groups = []
    for g in args.group:
        label, jobs_str = g.split(":", 1)
        job_ids = [j.strip() for j in jobs_str.split(",") if j.strip()]
        per_seed = []  # list of (seed, steps, smoothed_reward)
        for jid in job_ids:
            entries = find_train_csv(jid, args.root)
            if not entries:
                print(f"  {jid}: no train.csv found under {args.root}/{jid}")
                continue
            for seed, csv_path in entries:
                try:
                    steps, reward = load_curve(csv_path, smooth_window=args.smooth)
                except Exception as e:
                    print(f"  {jid}/seed_{seed}: skip ({e})")
                    continue
                per_seed.append((f"{jid}/seed_{seed}", steps, reward))

        if not per_seed:
            print(f"Group '{label}': no data, skipping")
            continue

        # Print step coverage per seed
        for tag, s, _ in per_seed:
            print(f"    {tag}: {len(s)} eps, step range {int(s[0])}..{int(s[-1])}")
        # Build shared grid spanning the FULL union range. Each seed only
        # contributes to grid points within its own step range; the rest are
        # NaN and ignored by nanmean/nanstd. This stops a single short
        # (timed-out) seed from clipping the plot.
        step_max = max(s[-1] for _, s, _ in per_seed)
        step_min = min(s[0] for _, s, _ in per_seed)
        grid = np.linspace(step_min, step_max, args.grid)

        rows = []
        for _, s, r in per_seed:
            interp = np.interp(grid, s, r, left=np.nan, right=np.nan)
            rows.append(interp)
        stacked = np.stack(rows)
        groups.append((label, grid, stacked, per_seed))
        last10_slice = stacked[:, int(args.grid * 0.9):]
        last10 = np.nanmean(last10_slice, axis=1)
        valid_mask = ~np.isnan(last10)
        print(f"  {label}: {len(per_seed)} seeds total, "
              f"{valid_mask.sum()} reach last-10% region. "
              f"last-10% per surviving seed = {last10[valid_mask].round(1).tolist()}, "
              f"mean = {np.nanmean(last10):.1f} ± {np.nanstd(last10):.1f}")

    fig, ax = plt.subplots(figsize=(11, 6))
    default_colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    if args.colors:
        colors = [c.strip() for c in args.colors.split(",")]
    else:
        colors = default_colors

    for i, (label, grid, stacked, per_seed) in enumerate(groups):
        c = colors[i % len(colors)]
        mean = np.nanmean(stacked, axis=0)
        std = np.nanstd(stacked, axis=0)
        last10_m = np.nanmean(stacked[:, int(len(grid) * 0.9):])
        last10_per_seed = np.nanmean(stacked[:, int(len(grid) * 0.9):], axis=1)
        last10_s = np.nanstd(last10_per_seed)
        if args.show_seeds:
            for _, s, r in per_seed:
                ax.plot(s, r, color=c, alpha=0.20, linewidth=0.7)
        ax.plot(grid / 1000.0, mean, color=c, linewidth=1.8,
                label=f"{label} (n={len(per_seed)} seeds, last10%={last10_m:.1f}±{last10_s:.1f})")
        ax.fill_between(grid / 1000.0, mean - std, mean + std, color=c, alpha=0.18)

    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_xlabel("Training Steps (x1000)")
    ax.set_ylabel("True Episode Reward (rolling mean across episodes)")
    ax.set_title(args.title)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    if args.ylim:
        ymin, ymax = (float(v) for v in args.ylim.split(","))
        ax.set_ylim(ymin, ymax)
    if args.xlim:
        xmin, xmax = (float(v) for v in args.xlim.split(","))
        ax.set_xlim(xmin, xmax)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
