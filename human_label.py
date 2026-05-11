#!/usr/bin/env python3
"""
Terminal-based human preference labeling for PEBBLE.

Standalone usage (offline labeling on saved trajectory data):
    python human_label.py --traj_file trajectories.pkl --output human_labels.pkl \
        [--num_queries 50] [--segment 50]

During training:
    Set feed_type=6 in your PEBBLE run script to get interactive prompts.
"""

import os
import sys
import numpy as np
import pickle
import argparse

# ---------------------------------------------------------------------------
# Environment-specific observation descriptions
# ---------------------------------------------------------------------------
ENV_OBS_LABELS = {
    'LunarLander': {
        'labels': ['x', 'y', 'vx', 'vy', 'angle', 'ang_vel', 'left_leg', 'right_leg'],
        'obs_dim': 8,
    },
}


def detect_env_type(obs_dim):
    """Guess the environment from observation dimensionality."""
    for name, info in ENV_OBS_LABELS.items():
        if info['obs_dim'] == obs_dim:
            return name
    return None


# ---------------------------------------------------------------------------
# Segment formatting
# ---------------------------------------------------------------------------

def _fmt_lunar(obs_start, obs_end, obs_all, act_all, length):
    """Pretty-print a LunarLander segment."""
    lines = []
    lines.append(f"  Steps: {length}")
    lines.append(f"  Pos:   ({obs_start[0]:+.3f}, {obs_start[1]:+.3f}) -> ({obs_end[0]:+.3f}, {obs_end[1]:+.3f})")
    lines.append(f"  Vel:   ({obs_start[2]:+.3f}, {obs_start[3]:+.3f}) -> ({obs_end[2]:+.3f}, {obs_end[3]:+.3f})")
    lines.append(f"  Angle: {obs_start[4]:+.3f} -> {obs_end[4]:+.3f} rad")
    lines.append(f"  AngV:  {obs_start[5]:+.3f} -> {obs_end[5]:+.3f}")
    l0, r0 = 'Y' if obs_start[6] > 0.5 else 'N', 'Y' if obs_start[7] > 0.5 else 'N'
    l1, r1 = 'Y' if obs_end[6] > 0.5 else 'N', 'Y' if obs_end[7] > 0.5 else 'N'
    lines.append(f"  Legs:  L={l0} R={r0} -> L={l1} R={r1}")
    y = obs_all[:, 1]
    lines.append(f"  Height range: [{y.min():.3f}, {y.max():.3f}]  final={obs_end[1]:.3f}")
    lines.append(f"  Thrust avg:   main={act_all[:, 0].mean():+.3f}  lat={act_all[:, 1].mean():+.3f}")
    return '\n'.join(lines)


def _fmt_generic(obs_start, obs_end, obs_all, act_all, length, obs_dim):
    """Pretty-print a generic segment."""
    lines = []
    lines.append(f"  Steps: {length}   obs_dim={obs_dim}  act_dim={act_all.shape[1]}")
    n = min(6, obs_dim)
    sfx = ', ...' if obs_dim > n else ''
    s_str = ', '.join(f'{v:+.3f}' for v in obs_start[:n])
    e_str = ', '.join(f'{v:+.3f}' for v in obs_end[:n])
    lines.append(f"  Start: [{s_str}{sfx}]")
    lines.append(f"  End:   [{e_str}{sfx}]")
    lines.append(f"  Obs range: [{obs_all.min():.3f}, {obs_all.max():.3f}]")
    lines.append(f"  Act range: [{act_all.min():.3f}, {act_all.max():.3f}]")
    return '\n'.join(lines)


def format_segment(segment, length, obs_dim, env_type=None):
    """Return a multi-line string describing one segment."""
    obs = segment[:length, :obs_dim]
    act = segment[:length, obs_dim:]
    obs_start, obs_end = obs[0], obs[length - 1]

    if env_type == 'LunarLander':
        return _fmt_lunar(obs_start, obs_end, obs, act, length)
    return _fmt_generic(obs_start, obs_end, obs, act, length, obs_dim)


# ---------------------------------------------------------------------------
# Interactive query presentation
# ---------------------------------------------------------------------------

def present_query(idx, total, seg1, seg2, len1, len2, obs_dim, env_type=None):
    """Show one preference query. Returns (label, quit_flag).

    Label convention (matches PEBBLE):
        0  = segment A is better  (seg1 preferred)
        1  = segment B is better  (seg2 preferred)
       -1  = equally good
       None = skip
    """
    print(f"\n{'=' * 60}")
    print(f"  Query {idx + 1} / {total}")
    print(f"{'=' * 60}")

    print(f"\n  --- Segment A ---")
    print(format_segment(seg1, len1, obs_dim, env_type))

    print(f"\n  --- Segment B ---")
    print(format_segment(seg2, len2, obs_dim, env_type))

    print(f"\n{'~' * 60}")
    print("  [1] A is better   [2] B is better   [0] Equal")
    print("  [s] Skip          [q] Quit & save")
    print(f"{'~' * 60}")

    while True:
        try:
            choice = input("  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return None, True
        if choice == '1':
            return 0, False      # seg1 preferred
        elif choice == '2':
            return 1, False      # seg2 preferred
        elif choice == '0':
            return -1, False     # equal
        elif choice == 's':
            return None, False   # skip
        elif choice == 'q':
            return None, True    # quit
        else:
            print("  Enter 1, 2, 0, s, or q.")


# ---------------------------------------------------------------------------
# Batch labeling API  (called by RewardModel.human_sampling)
# ---------------------------------------------------------------------------

def get_human_labels(sa_t_1, sa_t_2, len_1, len_2, obs_dim, env_type=None):
    """Present all queries and collect human labels.

    Returns
    -------
    labels : ndarray (n_labeled, 1)
    keep_indices : ndarray of ints — indices into the original batch that
                   were actually labeled (skipped queries excluded).
    """
    if env_type is None:
        env_type = detect_env_type(obs_dim)

    batch_size = sa_t_1.shape[0]
    labels = []
    keep_indices = []

    print(f"\n{'#' * 60}")
    print(f"  HUMAN PREFERENCE LABELING  —  {batch_size} queries")
    print(f"{'#' * 60}")

    for i in range(batch_size):
        label, quit_flag = present_query(
            i, batch_size,
            sa_t_1[i], sa_t_2[i],
            int(len_1[i]), int(len_2[i]),
            obs_dim, env_type,
        )

        if quit_flag:
            print(f"  Stopped early. Labeled {len(labels)} / {batch_size} queries.")
            break

        if label is not None:
            labels.append(label)
            keep_indices.append(i)

    if len(labels) == 0:
        return np.array([], dtype=np.float32).reshape(0, 1), np.array([], dtype=np.int64)

    return (np.array(labels, dtype=np.float32).reshape(-1, 1),
            np.array(keep_indices, dtype=np.int64))


# ---------------------------------------------------------------------------
# Standalone CLI — offline labeling on saved trajectories
# ---------------------------------------------------------------------------

def _sample_segments(inputs, targets, size_segment, num_queries):
    """Reproduce the segment-sampling logic from RewardModel.get_queries."""
    # Filter finished trajectories
    eligible = [(inp, tgt) for inp, tgt in zip(inputs, targets)
                if not isinstance(inp, list) and len(inp) > 0]
    if len(eligible) == 0:
        raise RuntimeError("No eligible trajectories found in the data.")

    sa_1, sa_2, r_1, r_2, l1, l2 = [], [], [], [], [], []

    for _ in range(num_queries):
        i1, i2 = np.random.randint(len(eligible), size=2)
        traj1, rew1 = eligible[i1]
        traj2, rew2 = eligible[i2]

        def _extract(traj, rew):
            if len(traj) >= size_segment:
                s = np.random.randint(0, len(traj) - size_segment + 1)
                return traj[s:s + size_segment], rew[s:s + size_segment], size_segment
            pad_sa = np.zeros((size_segment, traj.shape[1]), dtype=np.float32)
            pad_r  = np.zeros((size_segment, rew.shape[1]),  dtype=np.float32)
            pad_sa[:len(traj)] = traj
            pad_r[:len(rew)]   = rew
            return pad_sa, pad_r, len(traj)

        s1, rv1, ln1 = _extract(traj1, rew1)
        s2, rv2, ln2 = _extract(traj2, rew2)
        sa_1.append(s1); r_1.append(rv1); l1.append(ln1)
        sa_2.append(s2); r_2.append(rv2); l2.append(ln2)

    return (np.array(sa_1, dtype=np.float32),
            np.array(sa_2, dtype=np.float32),
            np.array(r_1,  dtype=np.float32),
            np.array(r_2,  dtype=np.float32),
            np.array(l1, dtype=np.int32),
            np.array(l2, dtype=np.int32))


# ---------------------------------------------------------------------------
# Video-based offline labeling from saved query batches
# ---------------------------------------------------------------------------

def _play_video_cv2(video_path, window_name='Segment'):
    """Play an mp4 video in a cv2 window. Returns True if played successfully."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [error] Cannot open {video_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    delay = max(1, int(1000 / fps))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow(window_name, frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cap.release()
    return True


def _play_pair_videos(seg_a_path, seg_b_path):
    """Play segment A then segment B videos sequentially."""
    import cv2
    print("  Playing Segment A...")
    _play_video_cv2(seg_a_path, 'Segment A')
    print("  Playing Segment B...")
    _play_video_cv2(seg_b_path, 'Segment B')
    cv2.destroyAllWindows()


def present_query_video(idx, total, seg_a_path, seg_b_path):
    """Show one video-based preference query. Returns (label, quit_flag).

    Label convention (matches PEBBLE):
        0  = segment A is better  (seg1 preferred)
        1  = segment B is better  (seg2 preferred)
       -1  = equally good
       None = skip
    """
    print(f"\n{'=' * 60}")
    print(f"  Query {idx + 1} / {total}")
    print(f"{'=' * 60}")

    _play_pair_videos(seg_a_path, seg_b_path)

    print(f"\n{'~' * 60}")
    print("  [1] A is better   [2] B is better   [0] Equal")
    print("  [s] Skip   [r] Replay videos   [q] Quit & save")
    print(f"{'~' * 60}")

    while True:
        try:
            choice = input("  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return None, True
        if choice == '1':
            return 0, False
        elif choice == '2':
            return 1, False
        elif choice == '0':
            return -1, False
        elif choice == 's':
            return None, False
        elif choice == 'r':
            _play_pair_videos(seg_a_path, seg_b_path)
        elif choice == 'q':
            return None, True
        else:
            print("  Enter 1, 2, 0, s, r, or q.")


def label_from_query_dir(query_dir, output_path, csv_path=None):
    """Load saved query batches and label them with video replay.

    Args:
        query_dir: directory containing batch_NNN/ subdirs with videos + metadata
        output_path: where to save human_labels.pkl
        csv_path: where to save the numerical CSV log (default: output_path + .csv)
    """
    import cv2
    import glob
    import time as _time

    if csv_path is None:
        csv_path = output_path.replace('.pkl', '') + '.csv'

    # Load batch metadata files
    index_path = os.path.join(query_dir, 'index.pkl')
    if os.path.exists(index_path):
        with open(index_path, 'rb') as f:
            batch_dirs = pickle.load(f)
    else:
        batch_dirs = sorted(glob.glob(os.path.join(query_dir, 'batch_*')))

    if not batch_dirs:
        print(f"No batch directories found in {query_dir}")
        return

    # Collect all query pairs across batches
    all_pairs = []  # list of (batch_idx, pair_idx, seg_a_path, seg_b_path, meta)
    for batch_dir in batch_dirs:
        meta_path = os.path.join(batch_dir, 'metadata.pkl')
        if not os.path.exists(meta_path):
            print(f"  [warn] skipping {batch_dir} — no metadata.pkl")
            continue
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)

        n_pairs = meta['n_pairs']
        batch_idx = meta['batch_idx']
        for i in range(n_pairs):
            pair_dir = os.path.join(batch_dir, f'pair_{i:03d}')
            seg_a = os.path.join(pair_dir, 'seg_A.mp4')
            seg_b = os.path.join(pair_dir, 'seg_B.mp4')
            if os.path.exists(seg_a) and os.path.exists(seg_b):
                all_pairs.append((batch_idx, i, seg_a, seg_b, meta))
            else:
                print(f"  [warn] missing videos for batch {batch_idx} pair {i}")

    total = len(all_pairs)
    if total == 0:
        print("No valid query pairs found.")
        return

    print(f"\n{'#' * 60}")
    print(f"  VIDEO PREFERENCE LABELING  —  {total} queries")
    print(f"{'#' * 60}")

    # Collect labels with timing
    labels_list = []
    keep_indices = []
    timing_rows = []  # for CSV
    cumulative_time = 0.0

    # Collect segment arrays for output pkl
    sa_1_list, sa_2_list, len_1_list, len_2_list = [], [], [], []
    oracle_labels_list = []

    for global_idx, (batch_idx, pair_idx, seg_a, seg_b, meta) in enumerate(all_pairs):
        t_start = _time.time()

        label, quit_flag = present_query_video(
            global_idx, total, seg_a, seg_b)

        t_end = _time.time()
        dt = t_end - t_start

        if quit_flag:
            print(f"  Stopped early. Labeled {len(labels_list)} / {total} queries.")
            break

        # Get oracle label for this pair
        oracle_label = None
        if meta.get('oracle_labels') is not None:
            oracle_label = float(meta['oracle_labels'][pair_idx].flat[0])

        if label is not None:
            cumulative_time += dt
            labels_list.append(label)
            keep_indices.append(global_idx)
            sa_1_list.append(meta['sa_t_1'][pair_idx])
            sa_2_list.append(meta['sa_t_2'][pair_idx])
            len_1_list.append(meta['len_1'][pair_idx])
            len_2_list.append(meta['len_2'][pair_idx])
            oracle_labels_list.append(oracle_label)

            timing_rows.append({
                'batch_idx': batch_idx,
                'query_idx': pair_idx,
                'label': label,
                'time_sec': round(dt, 3),
                'cumulative_time_sec': round(cumulative_time, 3),
                'oracle_label': oracle_label,
            })

    cv2.destroyAllWindows()

    if len(labels_list) == 0:
        print("No labels collected. Nothing saved.")
        return

    # Save pkl (compatible with reward_model.load_human_labels)
    labels_arr = np.array(labels_list, dtype=np.float32).reshape(-1, 1)
    out = {
        'sa_t_1': np.array(sa_1_list, dtype=np.float32),
        'sa_t_2': np.array(sa_2_list, dtype=np.float32),
        'labels': labels_arr,
        'len_1': np.array(len_1_list, dtype=np.int32),
        'len_2': np.array(len_2_list, dtype=np.int32),
        'obs_dim': meta['obs_dim'],
        'act_dim': meta['act_dim'],
        'size_segment': meta['size_segment'],
    }
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(out, f)
    print(f"\nSaved {len(labels_list)} labels -> {output_path}")

    # Save CSV numerical log
    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
    with open(csv_path, 'w') as f:
        f.write("batch_idx,query_idx,label,time_sec,cumulative_time_sec,oracle_label\n")
        for row in timing_rows:
            f.write(f"{row['batch_idx']},{row['query_idx']},{row['label']},"
                    f"{row['time_sec']},{row['cumulative_time_sec']},"
                    f"{row['oracle_label']}\n")
    print(f"Saved timing log -> {csv_path}")
    print(f"Total human labeling time: {cumulative_time:.1f} sec")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Offline human preference labeling for PEBBLE")

    # Mode 1: text-based labeling from saved trajectories (original)
    parser.add_argument('--traj_file', type=str, default=None,
                        help='Path to trajectories.pkl (text-based labeling mode)')

    # Mode 2: video-based labeling from saved query batches (new)
    parser.add_argument('--query_dir', type=str, default=None,
                        help='Path to query_videos/ directory (video labeling mode)')

    parser.add_argument('--output', type=str, default='human_labels.pkl',
                        help='Output path for labeled preferences')
    parser.add_argument('--csv', type=str, default=None,
                        help='Output path for timing CSV log (default: output.csv)')
    parser.add_argument('--num_queries', type=int, default=50,
                        help='Number of queries (text mode only)')
    parser.add_argument('--segment', type=int, default=None,
                        help='Segment length (text mode only)')
    args = parser.parse_args()

    if args.query_dir:
        # Video-based labeling mode
        label_from_query_dir(args.query_dir, args.output, args.csv)

    elif args.traj_file:
        # Original text-based labeling mode
        with open(args.traj_file, 'rb') as f:
            data = pickle.load(f)

        inputs = data['inputs']
        targets = data['targets']
        obs_dim = data['ds']
        act_dim = data['da']
        size_segment = args.segment or data['size_segment']
        env_type = detect_env_type(obs_dim)

        print(f"Loaded {len(inputs)} trajectories  (obs={obs_dim}, act={act_dim}, seg={size_segment})")
        if env_type:
            print(f"Detected environment: {env_type}")

        sa_t_1, sa_t_2, r_t_1, r_t_2, len_1, len_2 = _sample_segments(
            inputs, targets, size_segment, args.num_queries)

        labels, keep = get_human_labels(sa_t_1, sa_t_2, len_1, len_2, obs_dim, env_type)

        if len(labels) == 0:
            print("No labels collected. Nothing saved.")
            return

        out = {
            'sa_t_1': sa_t_1[keep],
            'sa_t_2': sa_t_2[keep],
            'labels': labels,
            'len_1': len_1[keep],
            'len_2': len_2[keep],
            'obs_dim': obs_dim,
            'act_dim': act_dim,
            'size_segment': size_segment,
        }

        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'wb') as f:
            pickle.dump(out, f)
        print(f"\nSaved {len(labels)} labels -> {args.output}")

    else:
        parser.error("Must specify either --query_dir (video mode) or --traj_file (text mode)")


if __name__ == '__main__':
    main()
