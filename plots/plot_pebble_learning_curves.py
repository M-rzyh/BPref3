"""PEBBLE learning curves: true-environment return vs ENVIRONMENT steps.

Synthetic (scripted-teacher) runs only for now; the human arm can be added as a
second panel once those runs exist.

WHAT THE AXES ARE (verified in code and on disk, not assumed):
  y  TRUE environment return. train.csv has BOTH `episode_reward` (the LEARNED
     reward the agent optimizes) and `true_episode_reward` (the environment's own
     reward, logged for analysis only, train_PEBBLE.py:1140). This script uses
     true_episode_reward. For the true-reward SAC baseline train_SAC.py logs only
     `episode_reward`, which there IS the environment reward, so the fallback is
     correct for that arm.
  x  ENVIRONMENT steps, the `step` column (one env step per agent step; PEBBLE is
     online, 1 gradient update per env step after the seeding phase).

Episodes are binned in env-step windows and averaged, then aggregated across
seeds as mean +/- s.e. Colours are the standard tab10 palette.

Run from ~/BPref3 with the bpref39_clone env:
    python plots/plot_pebble_learning_curves.py
    python plots/plot_pebble_learning_curves.py --bin 10000 --smooth 3 --err sd
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/scratch/marzii/compare_runs")
OUT_DEFAULT = Path("/scratch/marzii/compare_runs/pebble/figures/"
                   "pebble_learning_curves_synthetic.png")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--synthetic_root", type=Path, default=ROOT / "pebble/lunarlander")
    p.add_argument("--baseline_root", type=Path, default=ROOT / "sac_truereward/lunarlander",
                   help="true-reward SAC runs, drawn as a reference curve")
    p.add_argument("--no_baseline", action="store_true")
    p.add_argument("--bin", type=int, default=10000, help="env-step bin width")
    p.add_argument("--max_steps", type=int, default=1000000)
    p.add_argument("--min_steps", type=int, default=900000,
                   help="ignore runs that stopped before this step")
    # NOTE: job ids cannot order runs here -- the cluster counter was reset, so the
    # April/May runs have 7-digit ids (4596561) while the 2026-09 sweep has 6-digit
    # ones (973283). The sweep is therefore listed explicitly. Those older runs used
    # a 400-step cap and different budgets/batches, so they are NOT comparable.
    p.add_argument("--jobs", default="973283,973284,973285,973286,973287,973288,"
                                     "973289,973290,973291,973292,977480",
                   help="comma-separated Slurm job ids to include (the synthetic "
                        "sweep); pass an empty string to include everything found")
    p.add_argument("--smooth", type=int, default=3,
                   help="centred rolling mean over this many bins (1 = none)")
    p.add_argument("--err", choices=["se", "sd"], default="se")
    p.add_argument("--out", type=Path, default=OUT_DEFAULT)
    return p.parse_args()


def smooth(y, k):
    """Centred rolling mean, truncating at the edges (no zero padding)."""
    if k <= 1:
        return y
    h, n = k // 2, y.shape[-1]
    out = np.empty_like(y, dtype=float)
    for i in range(n):
        lo, hi = max(0, i - h), min(n, i + h + 1)
        out[..., i] = np.nanmean(y[..., lo:hi], axis=-1)
    return out


def run_curve(run_dir: Path, bin_w: int, max_steps: int):
    """(binned true return, final step, n_labels) for one run, or None."""
    f = run_dir / "train.csv"
    if not f.exists():
        return None
    steps, rets, feeds = [], [], []
    with open(f) as fh:
        for row in csv.DictReader(fh):
            try:
                steps.append(float(row["step"]))
                rets.append(float(row.get("true_episode_reward")
                                  or row.get("episode_reward")))
                feeds.append(float(row.get("total_feedback", 0) or 0))
            except (TypeError, ValueError):
                continue
    if not steps:
        return None
    n_bins = max_steps // bin_w
    sums, counts = np.zeros(n_bins), np.zeros(n_bins)
    idx = np.clip((np.asarray(steps) // bin_w).astype(int), 0, n_bins - 1)
    np.add.at(sums, idx, np.asarray(rets))
    np.add.at(counts, idx, 1)
    vals = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    for i in range(1, n_bins):            # forward-fill empty bins
        if np.isnan(vals[i]):
            vals[i] = vals[i - 1]
    return vals, max(steps), (int(max(feeds)) if feeds else 0)


def collect(root: Path, a, label):
    """{n_labels: [curves]} over every finished run under root."""
    by_n = {}
    if not root.exists():
        print("  (missing) %s" % root)
        return by_n
    for d in sorted(list(root.glob("*/seed_*/pebble")) + list(root.glob("*/seed_*"))):
        if not (d / "train.csv").exists():
            continue
        job = d.parts[-3] if d.name == "pebble" else d.parts[-2]
        if label != "sac-base" and a.jobs and job not in a.jobs.split(","):
            continue
        c = run_curve(d, a.bin, a.max_steps)
        if c is None:
            continue
        vals, final_step, n_labels = c
        if final_step < a.min_steps:
            print("  (unfinished %d steps) %s" % (final_step, d))
            continue
        by_n.setdefault(n_labels, []).append(vals)
        print("  %-9s N=%-5d final_step=%d  %s" % (label, n_labels, final_step, d))
    return by_n


def main():
    a = parse_args()
    palette = plt.get_cmap("tab10").colors

    print("synthetic runs:")
    syn = collect(a.synthetic_root, a, "synthetic")
    base = {} if a.no_baseline else collect(a.baseline_root, a, "sac-base")

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = (np.arange(a.max_steps // a.bin) + 1) * a.bin

    for i, n in enumerate(sorted(syn)):
        v = smooth(np.vstack(syn[n]), a.smooth)
        m = np.nanmean(v, axis=0)
        sd = np.nanstd(v, axis=0, ddof=1) if v.shape[0] > 1 else np.zeros_like(m)
        e = sd / np.sqrt(v.shape[0]) if a.err == "se" else sd
        col = palette[i % len(palette)]
        ax.plot(x, m, color=col, lw=1.7,
                label="synthetic, N=%d labels (n=%d seeds)" % (n, v.shape[0]))
        ax.fill_between(x, m - e, m + e, color=col, alpha=0.18, lw=0)
        print("  synthetic N=%d final %.1f" % (n, m[-1]))

    if base:
        allc = [c for v in base.values() for c in v]
        v = smooth(np.vstack(allc), a.smooth)
        m = np.nanmean(v, axis=0)
        ax.plot(x, m, color="0.35", ls="--", lw=1.6,
                label="true-reward SAC (n=%d seeds)" % v.shape[0])
        print("  true-reward SAC final %.1f" % m[-1])

    ax.set_xlabel("environment steps")
    ax.set_ylabel("true environment return")
    ax.grid(alpha=0.3)
    ax.axhline(0, color="0.8", lw=0.8, zorder=0)
    ax.legend(fontsize=8.5, loc="lower right")
    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=200)
    print("\nwrote %s" % a.out)


if __name__ == "__main__":
    main()
