"""PEBBLE: final performance vs number of preferences, human vs synthetic labels.

PEBBLE run dirs do not carry the label budget in their path, so N is read from
each run's own train.csv (the max of the `total_feedback` column). Arms are
distinguished by which root they live under:

    <synthetic_root>/<jobid>/seed_<s>/pebble/     scripted teacher (feed_type=0)
    <human_root>/<jobid>/seed_<s>/pebble/         online web labeling (feed_type=7)

Metric, in order of preference per run:
  1. final_eval_<K>ep.json  written by a post-hoc deterministic evaluation
     (mean_return) -- this is the number to prefer once you have it;
  2. else the mean true_episode_reward over the last --last_episodes training
     episodes in train.csv. That is the STOCHASTIC behaviour policy during
     training, not a clean evaluation -- the plot labels it as such.

Optionally draws the true-reward SAC baseline as a horizontal band.

Usage (compute node, bpref39_clone env):
    python plots/plot_pebble_count_axis.py
    python plots/plot_pebble_count_axis.py --last_episodes 50 --err sd
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/scratch/marzii/compare_runs")
OUT_DEFAULT = Path("/scratch/marzii/compare_runs/pebble/figures/"
                   "pebble_count_axis_human_vs_synthetic.png")

HUMAN_C = "#1f77b4"
SYNTH_C = "#d62728"
BASE_C = "#555555"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--synthetic_root", type=Path, default=ROOT / "pebble/lunarlander")
    p.add_argument("--human_root", type=Path, default=ROOT / "pebble/lunarlander_web_full")
    p.add_argument("--baseline_root", type=Path, default=ROOT / "sac_truereward/lunarlander",
                   help="true-reward SAC runs; drawn as a horizontal band")
    p.add_argument("--no_baseline", action="store_true")
    p.add_argument("--last_episodes", type=int, default=50,
                   help="fallback metric: mean true_episode_reward over the last K episodes")
    p.add_argument("--min_steps", type=int, default=900000,
                   help="ignore runs that did not reach this step count")
    # Job ids cannot order runs here: the cluster counter was reset, so the
    # April/May runs have 7-digit ids (4596561) while the 2026-09 sweep has
    # 6-digit ones (973283). Those older runs used a 400-step episode cap and
    # other budgets/batches, so they are NOT comparable and are excluded by
    # listing the sweep explicitly.
    p.add_argument("--jobs", default="973283,973284,973285,973286,973287,973288,"
                                     "973289,973290,973291,973292,977480",
                   help="comma-separated Slurm job ids for the synthetic sweep; "
                        "empty string = include every run found")
    p.add_argument("--err", choices=["se", "sd"], default="se")
    p.add_argument("--xscale", choices=["linear", "log"], default="linear",
                   help="linear starts the label axis at 0; log spreads the small budgets")
    p.add_argument("--out", type=Path, default=OUT_DEFAULT)
    p.add_argument("--title", default="PEBBLE on LunarLander: human vs synthetic preferences")
    return p.parse_args()


def read_train_csv(run_dir: Path):
    """Returns (steps, true_returns, total_feedback) as lists; empty on failure."""
    f = run_dir / "train.csv"
    if not f.exists():
        return [], [], []
    steps, rets, feeds = [], [], []
    with open(f) as fh:
        for row in csv.DictReader(fh):
            try:
                steps.append(float(row["step"]))
                rets.append(float(row.get("true_episode_reward", row.get("episode_reward", "nan"))))
                feeds.append(float(row.get("total_feedback", 0) or 0))
            except (TypeError, ValueError):
                continue
    return steps, rets, feeds


def run_metric(run_dir: Path, last_episodes: int):
    """(N_labels, value, source) for one run dir, or None if unusable."""
    steps, rets, feeds = read_train_csv(run_dir)
    if not steps:
        return None
    n_labels = int(max(feeds)) if feeds else 0
    final_steps = max(steps)

    # 1. a proper post-hoc evaluation, if it exists
    for j in sorted(run_dir.glob("final_eval_*ep.json")):
        try:
            return n_labels, float(json.load(open(j))["mean_return"]), "eval", final_steps
        except (KeyError, json.JSONDecodeError):
            pass

    # 2. fallback: last-K training episodes (stochastic policy)
    tail = [r for r in rets[-last_episodes:] if r == r]  # drop NaN
    if not tail:
        return None
    return n_labels, float(np.mean(tail)), "train_tail", final_steps


def collect(root: Path, last_episodes: int, min_steps: int, label: str, jobs=None):
    """{N: [values]} over every run dir under root."""
    by_n, sources = {}, set()
    if not root.exists():
        print(f"  (missing root) {root}")
        return by_n, sources
    # PEBBLE runs are <job>/seed_<s>/pebble or <job>/pebble; the true-reward SAC
    # baseline writes <job>/seed_<s> directly (no pebble/ level).
    cands = (sorted(root.glob("*/seed_*/pebble")) + sorted(root.glob("*/pebble"))
             + [d for d in sorted(root.glob("*/seed_*")) if (d / "train.csv").exists()])
    seen = set()
    for pebble_dir in cands:
        if pebble_dir in seen:
            continue
        seen.add(pebble_dir)
        parts = pebble_dir.parts
        job = parts[-3] if pebble_dir.name == "pebble" and parts[-2].startswith("seed_") else parts[-2]
        if jobs and label != "sac-base" and job not in jobs:
            continue
        m = run_metric(pebble_dir, last_episodes)
        if m is None:
            continue
        n_labels, val, src, final_steps = m
        if final_steps < min_steps:
            print(f"  (unfinished, {int(final_steps)} steps) {pebble_dir}")
            continue
        by_n.setdefault(n_labels, []).append(val)
        sources.add(src)
        print(f"  {label:<9} N={n_labels:<5} {val:8.1f}  [{src}]  {pebble_dir}")
    return by_n, sources


def series(by_n, err_kind):
    xs = sorted(by_n)
    m = np.array([np.mean(by_n[x]) for x in xs])
    sd = np.array([np.std(by_n[x], ddof=1) if len(by_n[x]) > 1 else 0.0 for x in xs])
    n = np.array([len(by_n[x]) for x in xs])
    e = sd / np.sqrt(n) if err_kind == "se" else sd
    return np.array(xs, dtype=float), m, e, n


def main():
    a = parse_args()

    print("synthetic runs:")
    jobs = set(a.jobs.split(",")) if a.jobs else None
    syn, syn_src = collect(a.synthetic_root, a.last_episodes, a.min_steps, "synthetic", jobs)
    print("human runs:")
    hum, hum_src = collect(a.human_root, a.last_episodes, a.min_steps, "human", jobs)

    if not syn and not hum:
        raise SystemExit("no usable runs found")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    if not a.no_baseline:
        base, _ = collect(a.baseline_root, a.last_episodes, a.min_steps, "sac-base", None)
        vals = [v for vs in base.values() for v in vs]
        if vals:
            bm = float(np.mean(vals))
            bsd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            be = bsd / np.sqrt(len(vals)) if a.err == "se" else bsd
            ax.axhline(bm, color=BASE_C, ls="--", lw=1.4, zorder=1)
            ax.axhspan(bm - be, bm + be, color=BASE_C, alpha=0.15, zorder=0)
            ax.text(0.02, 0.96, f"true-reward SAC ({bm:.0f}, n={len(vals)})",
                    transform=ax.transAxes, color=BASE_C, fontsize=8, va="top")

    for by_n, color, marker, ls, name in ((hum, HUMAN_C, "o", "-", "human preferences"),
                                          (syn, SYNTH_C, "s", "--", "synthetic preferences")):
        if not by_n:
            continue
        x, m, e, n = series(by_n, a.err)
        ax.errorbar(x, m, yerr=e, marker=marker, color=color, ls=ls, lw=2, capsize=3,
                    label=f"{name} (n={n.max()} seeds)")

    metric_note = ("deterministic eval" if "eval" in (syn_src | hum_src)
                   else f"mean of last {a.last_episodes} TRAINING episodes (stochastic policy)")
    all_x = sorted(set(list(syn) + list(hum)))
    if a.xscale == "log":
        ax.set_xscale("log")
        if all_x:
            ax.set_xticks(all_x)
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    else:
        if all_x:
            ax.set_xlim(0, max(all_x) * 1.04)   # linear axis anchored at 0
            ax.set_xticks(all_x)
    ax.set_xlabel("number of preference labels (N)")
    ax.set_ylabel(f"final return\n({metric_note})", fontsize=9)
    ax.set_title(a.title, fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()

    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=200)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
