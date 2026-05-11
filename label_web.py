#!/usr/bin/env python3
"""
Web-based human preference labeling for PEBBLE segment videos.

Usage:
    python label_web.py --query_dir /scratch/marzii/compare_runs/pebble/lunarlander/JOBID/query_videos \
                        --output human_labels.pkl \
                        --port 8080

Opens a browser UI showing segment A and B videos side by side.
Labels are saved to a pickle file compatible with reward_model.load_human_labels.
"""

import os
import sys
import glob
import pickle
import time
import argparse
import numpy as np
from flask import Flask, send_file, jsonify, request as flask_request

# ---------------------------------------------------------------------------
# Load query data
# ---------------------------------------------------------------------------

def load_all_pairs(query_dir):
    """Scan batch directories and collect all video pairs with metadata."""
    index_path = os.path.join(query_dir, 'index.pkl')
    if os.path.exists(index_path):
        with open(index_path, 'rb') as f:
            batch_dirs = pickle.load(f)
    else:
        batch_dirs = sorted(glob.glob(os.path.join(query_dir, 'batch_*')))

    pairs = []
    for batch_dir in batch_dirs:
        meta_path = os.path.join(batch_dir, 'metadata.pkl')
        if not os.path.exists(meta_path):
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
                pairs.append({
                    'batch_idx': batch_idx,
                    'pair_idx': i,
                    'seg_a': seg_a,
                    'seg_b': seg_b,
                    'meta': meta,
                })
    return pairs


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)

ALL_PAIRS = []
QUERY_DIR = ''
LABELS = []       # list of dicts: {pair_index, label, time_sec, ...}
# Set of (batch_idx, pair_idx) already labeled — survives refresh/rescan
LABELED_KEYS = set()
OUTPUT_PATH = 'human_labels.pkl'
CSV_PATH = None

HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>PEBBLE Preference Labeling</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #0f172a; color: #e2e8f0; min-height: 100vh; }
.header { text-align: center; padding: 20px; background: #1e293b; }
.header h1 { font-size: 22px; color: #f8fafc; }
.progress { font-size: 14px; color: #94a3b8; margin-top: 6px; }
.main { max-width: 1100px; margin: 20px auto; padding: 0 20px; }
.videos { display: flex; gap: 24px; justify-content: center; }
.video-panel { flex: 1; max-width: 520px; }
.video-panel h2 { text-align: center; font-size: 18px; margin-bottom: 10px;
                   padding: 8px; border-radius: 8px; }
.panel-a h2 { background: #1e3a5f; color: #60a5fa; }
.panel-b h2 { background: #3b1f3b; color: #c084fc; }
video { width: 100%; border-radius: 8px; background: #000; }
.controls { display: flex; gap: 12px; justify-content: center; margin: 24px 0; flex-wrap: wrap; }
.btn { padding: 14px 32px; border: 2px solid transparent; border-radius: 10px;
       font-size: 16px; font-weight: 600; cursor: pointer; transition: all 0.15s; }
.btn:hover { transform: translateY(-2px); }
.btn-a { background: #1e3a5f; color: #60a5fa; border-color: #2563eb; }
.btn-a:hover { background: #2563eb; color: #fff; }
.btn-b { background: #3b1f3b; color: #c084fc; border-color: #7c3aed; }
.btn-b:hover { background: #7c3aed; color: #fff; }
.btn-eq { background: #1e293b; color: #94a3b8; border-color: #475569; }
.btn-eq:hover { background: #475569; color: #fff; }
.btn-skip { background: #1e293b; color: #64748b; border-color: #334155; }
.btn-skip:hover { background: #334155; color: #fff; }
.btn-replay { background: #0f172a; color: #38bdf8; border-color: #0284c7; padding: 10px 20px; font-size: 14px; }
.btn-replay:hover { background: #0284c7; color: #fff; }
.btn-pause { background: #1e293b; color: #fbbf24; border-color: #d97706; padding: 10px 20px; font-size: 14px; }
.btn-pause:hover { background: #d97706; color: #fff; }
.btn-pause.paused { background: #d97706; color: #fff; }
.btn-refresh { background: #1e293b; color: #34d399; border-color: #059669; padding: 10px 20px; font-size: 14px; }
.btn-refresh:hover { background: #059669; color: #fff; }
.replay-row { display: flex; gap: 12px; justify-content: center; margin: 12px 0; }
.toolbar { display: flex; gap: 12px; justify-content: center; margin: 16px 0; }
.feedback { text-align: center; font-size: 16px; margin: 10px 0; min-height: 24px; color: #22c55e; }
.timer { text-align: center; font-size: 14px; color: #64748b; margin: 6px 0; }
.paused-banner { text-align: center; font-size: 18px; color: #fbbf24; margin: 10px 0;
                  padding: 10px; background: #422006; border-radius: 8px; display: none; }
.stats { text-align: center; font-size: 13px; color: #475569; margin-top: 20px; padding-top: 16px; border-top: 1px solid #1e293b; }
.done-screen { text-align: center; padding: 60px 20px; }
.done-screen h2 { font-size: 28px; color: #22c55e; margin-bottom: 12px; }
.done-screen .btn-refresh { margin-top: 16px; padding: 14px 32px; font-size: 16px; }
.keyboard-hint { text-align: center; font-size: 13px; color: #475569; margin-top: 8px; }
</style>
</head>
<body>
<div class="header">
  <h1>PEBBLE Preference Labeling - LunarLander</h1>
  <div class="progress" id="progress">Loading...</div>
</div>
<div class="main">
  <div id="labeling-ui">
    <div class="videos">
      <div class="video-panel panel-a">
        <h2>Segment A</h2>
        <video id="vid-a" muted></video>
      </div>
      <div class="video-panel panel-b">
        <h2>Segment B</h2>
        <video id="vid-b" muted></video>
      </div>
    </div>
    <div class="replay-row">
      <button class="btn btn-replay" onclick="replay('a')">Replay A</button>
      <button class="btn btn-replay" onclick="replay('b')">Replay B</button>
      <button class="btn btn-replay" onclick="replay('both')">Replay Both</button>
    </div>
    <div class="toolbar">
      <button class="btn btn-pause" id="pause-btn" onclick="togglePause()">Pause (p)</button>
      <button class="btn btn-refresh" onclick="doRefresh()">Load new segments (F5)</button>
    </div>
    <div class="paused-banner" id="paused-banner">PAUSED - timer stopped. Press P or click Pause to resume.</div>
    <div class="timer" id="timer"></div>
    <div class="controls" id="choice-controls">
      <button class="btn btn-a" onclick="choose(0)">A is better (1)</button>
      <button class="btn btn-eq" onclick="choose(-1)">Equal (0)</button>
      <button class="btn btn-b" onclick="choose(1)">B is better (2)</button>
      <button class="btn btn-skip" onclick="choose(null)">Skip (s)</button>
    </div>
    <div class="keyboard-hint">Keyboard: 1=A, 2=B, 0=Equal, s=Skip, r=Replay, p=Pause, F5=Load new</div>
    <div class="feedback" id="feedback"></div>
  </div>
  <div id="done-screen" style="display:none" class="done-screen">
    <h2>Caught up!</h2>
    <p id="done-msg"></p>
    <button class="btn btn-refresh" onclick="doRefresh()">Load new segments</button>
  </div>
  <div class="stats" id="stats"></div>
</div>
<script>
let idx = 0, total = 0, labeled = 0, cumTime = 0;
let startTime = null, pausedAt = null, pausedElapsed = 0, paused = false;

async function init() {
  const r = await fetch('/api/status');
  const d = await r.json();
  total = d.total;
  labeled = d.labeled;
  idx = d.current;
  cumTime = d.cumulative_time;
  paused = false;
  pausedAt = null;
  pausedElapsed = 0;
  document.getElementById('paused-banner').style.display = 'none';
  document.getElementById('pause-btn').classList.remove('paused');
  document.getElementById('pause-btn').textContent = 'Pause (p)';
  updateStats();
  if (idx < total) {
    document.getElementById('labeling-ui').style.display = '';
    document.getElementById('done-screen').style.display = 'none';
    loadPair(idx);
  } else {
    showDone();
  }
}

function updateStats() {
  document.getElementById('progress').textContent =
    `Query ${Math.min(idx+1, total)} / ${total}  |  Labeled: ${labeled}`;
  document.getElementById('stats').textContent =
    `Labeled: ${labeled}  |  Total time: ${cumTime.toFixed(1)}s`;
}

function loadPair(i) {
  const va = document.getElementById('vid-a');
  const vb = document.getElementById('vid-b');
  va.src = '/video/' + i + '/a?' + Date.now();
  vb.src = '/video/' + i + '/b?' + Date.now();
  va.load(); vb.load();
  let loadedCount = 0;
  function onLoaded() {
    loadedCount++;
    if (loadedCount === 2 && !paused) { va.play(); vb.play(); }
  }
  va.onloadeddata = onLoaded;
  vb.onloadeddata = onLoaded;
  startTime = performance.now();
  pausedElapsed = 0;
  document.getElementById('feedback').textContent = '';
  updateStats();
  updateTimer();
}

function getElapsed() {
  if (!startTime) return 0;
  if (paused && pausedAt) return pausedElapsed + (pausedAt - startTime) / 1000;
  return pausedElapsed + (performance.now() - startTime) / 1000;
}

function updateTimer() {
  if (idx < total) {
    const el = getElapsed();
    const label = paused ? 'PAUSED' : 'Decision time';
    document.getElementById('timer').textContent = `${label}: ${el.toFixed(1)}s`;
    if (!paused) requestAnimationFrame(updateTimer);
  }
}

function togglePause() {
  const va = document.getElementById('vid-a');
  const vb = document.getElementById('vid-b');
  if (!paused) {
    paused = true;
    pausedAt = performance.now();
    va.pause(); vb.pause();
    document.getElementById('paused-banner').style.display = '';
    document.getElementById('pause-btn').classList.add('paused');
    document.getElementById('pause-btn').textContent = 'Resume (p)';
    updateTimer();
  } else {
    paused = false;
    pausedElapsed += (performance.now() - pausedAt) / 1000;
    // Shift startTime forward by the paused duration so getElapsed stays correct
    startTime += (performance.now() - pausedAt);
    pausedAt = null;
    va.play(); vb.play();
    document.getElementById('paused-banner').style.display = 'none';
    document.getElementById('pause-btn').classList.remove('paused');
    document.getElementById('pause-btn').textContent = 'Pause (p)';
    updateTimer();
  }
}

async function doRefresh() {
  document.getElementById('feedback').textContent = 'Scanning for new segments...';
  const r = await fetch('/api/refresh', {method: 'POST'});
  const d = await r.json();
  total = d.total;
  labeled = d.labeled;
  idx = d.current;
  cumTime = d.cumulative_time;
  const added = d.new_pairs;
  document.getElementById('feedback').textContent =
    added > 0 ? `Found ${added} new pairs! (${total} total)` : 'No new segments yet.';
  paused = false;
  pausedAt = null;
  pausedElapsed = 0;
  document.getElementById('paused-banner').style.display = 'none';
  document.getElementById('pause-btn').classList.remove('paused');
  document.getElementById('pause-btn').textContent = 'Pause (p)';
  updateStats();
  if (idx < total) {
    document.getElementById('labeling-ui').style.display = '';
    document.getElementById('done-screen').style.display = 'none';
    loadPair(idx);
  } else {
    showDone();
  }
}

function replay(which) {
  if (paused) return;
  const va = document.getElementById('vid-a');
  const vb = document.getElementById('vid-b');
  if (which === 'a' || which === 'both') { va.currentTime = 0; va.play(); }
  if (which === 'b' || which === 'both') { vb.currentTime = 0; vb.play(); }
}

async function choose(label) {
  if (idx >= total || paused) return;
  const dt = getElapsed() - pausedElapsed;
  if (label !== null) { cumTime += dt; labeled++; }

  const r = await fetch('/api/label', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({pair_index: idx, label: label, time_sec: dt})
  });
  const d = await r.json();

  if (label !== null) {
    const names = {0: 'A preferred', 1: 'B preferred', '-1': 'Equal'};
    document.getElementById('feedback').textContent =
      (names[String(label)] || 'Labeled') + ` (${dt.toFixed(1)}s)`;
  } else {
    document.getElementById('feedback').textContent = 'Skipped';
  }

  idx = d.current;
  updateStats();
  if (idx < total) {
    setTimeout(() => loadPair(idx), 400);
  } else {
    showDone();
  }
}

function showDone() {
  document.getElementById('labeling-ui').style.display = 'none';
  document.getElementById('done-screen').style.display = 'block';
  document.getElementById('done-msg').textContent =
    `Labeled ${labeled} / ${total} pairs in ${cumTime.toFixed(1)}s. Labels saved. Click below to check for new segments.`;
}

document.addEventListener('keydown', e => {
  if (e.key === '1') choose(0);
  else if (e.key === '2') choose(1);
  else if (e.key === '0') choose(-1);
  else if (e.key === 's' || e.key === 'S') choose(null);
  else if (e.key === 'r' || e.key === 'R') replay('both');
  else if (e.key === 'p' || e.key === 'P') togglePause();
  else if (e.key === 'F5') { e.preventDefault(); doRefresh(); }
});

init();
</script>
</body>
</html>"""


@app.route('/')
def index():
    return HTML_PAGE


@app.route('/video/<int:pair_idx>/<segment>')
def serve_video(pair_idx, segment):
    if pair_idx < 0 or pair_idx >= len(ALL_PAIRS):
        return 'Not found', 404
    pair = ALL_PAIRS[pair_idx]
    path = pair['seg_a'] if segment == 'a' else pair['seg_b']
    return send_file(path, mimetype='video/mp4')


@app.route('/api/status')
def api_status():
    current = len(LABELS)
    labeled_count = sum(1 for l in LABELS if l['label'] is not None)
    cum_time = sum(l['time_sec'] for l in LABELS if l['label'] is not None)
    return jsonify({
        'total': len(ALL_PAIRS),
        'current': current,
        'labeled': labeled_count,
        'cumulative_time': round(cum_time, 3),
    })


@app.route('/api/refresh', methods=['POST'])
def api_refresh():
    """Rescan query_dir for new batches. Existing labels are preserved."""
    global ALL_PAIRS
    old_count = len(ALL_PAIRS)
    ALL_PAIRS = load_all_pairs(QUERY_DIR)
    new_count = len(ALL_PAIRS)
    added = new_count - old_count
    if added > 0:
        print(f"[refresh] found {added} new pairs ({new_count} total)")

    current = len(LABELS)
    labeled_count = sum(1 for l in LABELS if l['label'] is not None)
    cum_time = sum(l['time_sec'] for l in LABELS if l['label'] is not None)
    return jsonify({
        'total': new_count,
        'current': current,
        'labeled': labeled_count,
        'new_pairs': added,
        'cumulative_time': round(cum_time, 3),
    })


@app.route('/api/label', methods=['POST'])
def api_label():
    data = flask_request.get_json()
    pair_idx = data['pair_index']
    label = data['label']
    time_sec = data.get('time_sec', 0)

    pair = ALL_PAIRS[pair_idx]
    LABELS.append({
        'pair_index': pair_idx,
        'batch_idx': pair['batch_idx'],
        'pair_idx_in_batch': pair['pair_idx'],
        'label': label,
        'time_sec': round(time_sec, 3),
    })

    # Auto-save after every label
    _save_labels()

    labeled_count = sum(1 for l in LABELS if l['label'] is not None)
    cum_time = sum(l['time_sec'] for l in LABELS if l['label'] is not None)
    return jsonify({
        'ok': True,
        'current': len(LABELS),
        'labeled': labeled_count,
        'cumulative_time': round(cum_time, 3),
    })


def _save_labels():
    """Save labels in PEBBLE-compatible pickle format."""
    labeled = [l for l in LABELS if l['label'] is not None]
    if not labeled:
        return

    sa_1_list, sa_2_list, len_1_list, len_2_list = [], [], [], []
    labels_list = []

    for l in labeled:
        pair = ALL_PAIRS[l['pair_index']]
        meta = pair['meta']
        pi = pair['pair_idx']
        sa_1_list.append(meta['sa_t_1'][pi])
        sa_2_list.append(meta['sa_t_2'][pi])
        len_1_list.append(meta['len_1'][pi])
        len_2_list.append(meta['len_2'][pi])
        labels_list.append(l['label'])

    sample_meta = ALL_PAIRS[0]['meta']
    out = {
        'sa_t_1': np.array(sa_1_list, dtype=np.float32),
        'sa_t_2': np.array(sa_2_list, dtype=np.float32),
        'labels': np.array(labels_list, dtype=np.float32).reshape(-1, 1),
        'len_1': np.array(len_1_list, dtype=np.int32),
        'len_2': np.array(len_2_list, dtype=np.int32),
        'obs_dim': sample_meta['obs_dim'],
        'act_dim': sample_meta['act_dim'],
        'size_segment': sample_meta['size_segment'],
    }
    os.makedirs(os.path.dirname(OUTPUT_PATH) or '.', exist_ok=True)
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(out, f)

    # Also save CSV log
    csv_path = CSV_PATH or OUTPUT_PATH.replace('.pkl', '.csv')
    with open(csv_path, 'w') as f:
        f.write("pair_index,batch_idx,pair_idx,label,time_sec\n")
        for l in LABELS:
            f.write(f"{l['pair_index']},{l['batch_idx']},{l['pair_idx_in_batch']},"
                    f"{l['label']},{l['time_sec']}\n")


## -----------------------------------------------------------------------
## Online mode: watcher + auto response writing
## -----------------------------------------------------------------------

ONLINE_MODE = False
ONLINE_QUERY_DIR = ''
ONLINE_BATCH_LABELS = {}  # batch_dir -> list of label dicts
ONLINE_CURRENT_BATCH = None  # batch_dir currently being labeled
ONLINE_BATCH_QUEUE = []  # list of batch_dirs waiting for labels
# Per-batch timing for measuring "web open but human hasn't pressed Play yet".
# Set when a batch becomes ONLINE_CURRENT_BATCH and when the user first hits Play.
BATCH_ARRIVED_AT = {}     # batch_dir -> wall-clock seconds (time.time())
BATCH_FIRST_PLAY_AT = {}  # batch_dir -> wall-clock seconds (time.time())


def _scan_online_batches():
    """Walk ONLINE_QUERY_DIR recursively for batch_*/ dirs with request.json
    but no response.pkl. Supports both flat layout and parent-of-seeds layout
    (parent/seed_X/online_queries/batch_NNN/)."""
    import json
    new_batches = []
    if not os.path.isdir(ONLINE_QUERY_DIR):
        return new_batches
    for root, dirs, files in os.walk(ONLINE_QUERY_DIR):
        if 'request.json' in files and 'response.pkl' not in files:
            if (root not in ONLINE_BATCH_QUEUE) and (root != ONLINE_CURRENT_BATCH):
                new_batches.append(root)
    new_batches.sort()  # deterministic order across seeds
    return new_batches


def _load_online_batch(batch_dir):
    """Load pairs from an online batch directory."""
    meta_path = os.path.join(batch_dir, 'metadata.pkl')
    if not os.path.exists(meta_path):
        return []
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    pairs = []
    n = meta['n_pairs']
    batch_idx = meta['batch_idx']
    for i in range(n):
        pair_dir = os.path.join(batch_dir, f'pair_{i:03d}')
        seg_a = os.path.join(pair_dir, 'seg_A.mp4')
        seg_b = os.path.join(pair_dir, 'seg_B.mp4')
        if os.path.exists(seg_a) and os.path.exists(seg_b):
            pairs.append({
                'batch_idx': batch_idx,
                'pair_idx': i,
                'seg_a': seg_a,
                'seg_b': seg_b,
                'meta': meta,
                'batch_dir': batch_dir,
            })
    return pairs


def _write_online_response(batch_dir):
    """Write response.pkl for the training process to pick up."""
    labels_list = ONLINE_BATCH_LABELS.get(batch_dir, [])
    labeled = [l for l in labels_list if l['label'] is not None]

    # Wait-for-Play = time from batch arrival until first Play click.
    arrived_at = BATCH_ARRIVED_AT.get(batch_dir)
    first_play_at = BATCH_FIRST_PLAY_AT.get(batch_dir)
    if arrived_at is not None and first_play_at is not None:
        wait_for_start_sec = max(0.0, first_play_at - arrived_at)
    else:
        wait_for_start_sec = 0.0

    if not labeled:
        # Write empty response so training doesn't hang
        response = {
            'labels': np.array([], dtype=np.float32).reshape(0, 1),
            'keep_indices': np.array([], dtype=np.int64),
            'time_sec': 0.0,
            'wait_for_start_sec': wait_for_start_sec,
            'timestamp': time.time(),
            'n_labeled': 0,
            'n_skipped': len(labels_list),
        }
    else:
        response = {
            'labels': np.array([l['label'] for l in labeled], dtype=np.float32).reshape(-1, 1),
            'keep_indices': np.array([l['pair_idx'] for l in labeled], dtype=np.int64),
            'time_sec': sum(l['time_sec'] for l in labeled),
            'wait_for_start_sec': wait_for_start_sec,
            'timestamp': time.time(),
            'n_labeled': len(labeled),
            'n_skipped': len(labels_list) - len(labeled),
        }

    resp_path = os.path.join(batch_dir, 'response.pkl')
    tmp_path = resp_path + '.tmp'
    with open(tmp_path, 'wb') as f:
        pickle.dump(response, f)
    os.rename(tmp_path, resp_path)
    print(f"[online] wrote response for {os.path.basename(batch_dir)}: "
          f"{response['n_labeled']} labeled, {response['n_skipped']} skipped, "
          f"decision={response['time_sec']:.1f}s  "
          f"wait_for_start={wait_for_start_sec:.1f}s")


@app.route('/api/online/started', methods=['POST'])
def api_online_started():
    """Called by the browser the FIRST time Play is pressed for a batch.
    Records the timestamp so we can measure wait-for-start (web open but
    human hadn't started yet)."""
    if ONLINE_CURRENT_BATCH and ONLINE_CURRENT_BATCH not in BATCH_FIRST_PLAY_AT:
        BATCH_FIRST_PLAY_AT[ONLINE_CURRENT_BATCH] = time.time()
        wait = BATCH_FIRST_PLAY_AT[ONLINE_CURRENT_BATCH] \
               - BATCH_ARRIVED_AT.get(ONLINE_CURRENT_BATCH, BATCH_FIRST_PLAY_AT[ONLINE_CURRENT_BATCH])
        print(f"[online] human pressed Play for {os.path.basename(ONLINE_CURRENT_BATCH)} "
              f"after {wait:.1f}s wait")
    return jsonify({'ok': True})


@app.route('/api/online/status')
def api_online_status():
    """Return online mode state."""
    # Check for new batches
    new = _scan_online_batches()
    for b in new:
        if b not in ONLINE_BATCH_QUEUE:
            ONLINE_BATCH_QUEUE.append(b)

    current_pairs = []
    current_batch_name = None
    current_label_idx = 0

    global ONLINE_CURRENT_BATCH
    # If no current batch, pop from queue
    if ONLINE_CURRENT_BATCH is None and ONLINE_BATCH_QUEUE:
        ONLINE_CURRENT_BATCH = ONLINE_BATCH_QUEUE.pop(0)
        ONLINE_BATCH_LABELS[ONLINE_CURRENT_BATCH] = []
        BATCH_ARRIVED_AT[ONLINE_CURRENT_BATCH] = time.time()
        print(f"[online] starting batch: {os.path.basename(ONLINE_CURRENT_BATCH)}")

    if ONLINE_CURRENT_BATCH:
        current_pairs = _load_online_batch(ONLINE_CURRENT_BATCH)
        current_batch_name = os.path.basename(ONLINE_CURRENT_BATCH)
        current_label_idx = len(ONLINE_BATCH_LABELS.get(ONLINE_CURRENT_BATCH, []))

    return jsonify({
        'online': True,
        'has_batch': ONLINE_CURRENT_BATCH is not None,
        'batch_name': current_batch_name,
        'total_pairs': len(current_pairs),
        'current_idx': current_label_idx,
        'queue_size': len(ONLINE_BATCH_QUEUE),
        'labeled_total': sum(len(v) for v in ONLINE_BATCH_LABELS.values()),
    })


@app.route('/api/online/label', methods=['POST'])
def api_online_label():
    """Submit a label for the current online batch pair."""
    global ONLINE_CURRENT_BATCH

    data = flask_request.get_json()
    pair_idx = data['pair_index']
    label = data['label']
    time_sec = data.get('time_sec', 0)

    if ONLINE_CURRENT_BATCH is None:
        return jsonify({'ok': False, 'error': 'no active batch'}), 400

    batch_labels = ONLINE_BATCH_LABELS.setdefault(ONLINE_CURRENT_BATCH, [])
    batch_labels.append({
        'pair_idx': pair_idx,
        'label': label,
        'time_sec': round(time_sec, 3),
    })

    # Check if batch is complete
    pairs = _load_online_batch(ONLINE_CURRENT_BATCH)
    batch_done = len(batch_labels) >= len(pairs)

    if batch_done:
        _write_online_response(ONLINE_CURRENT_BATCH)
        _save_labels()  # also save cumulative offline-style labels
        finished_batch = os.path.basename(ONLINE_CURRENT_BATCH)
        ONLINE_CURRENT_BATCH = None

        # Auto-pick next batch if available
        new = _scan_online_batches()
        for b in new:
            if b not in ONLINE_BATCH_QUEUE:
                ONLINE_BATCH_QUEUE.append(b)
        if ONLINE_BATCH_QUEUE:
            ONLINE_CURRENT_BATCH = ONLINE_BATCH_QUEUE.pop(0)
            ONLINE_BATCH_LABELS[ONLINE_CURRENT_BATCH] = []
            BATCH_ARRIVED_AT[ONLINE_CURRENT_BATCH] = time.time()
            print(f"[online] starting batch: {os.path.basename(ONLINE_CURRENT_BATCH)}")

        return jsonify({
            'ok': True,
            'batch_done': True,
            'finished_batch': finished_batch,
            'has_next': ONLINE_CURRENT_BATCH is not None,
            'current_idx': 0,
        })

    return jsonify({
        'ok': True,
        'batch_done': False,
        'current_idx': len(batch_labels),
    })


@app.route('/online_video/<int:pair_idx>/<segment>')
def serve_online_video(pair_idx, segment):
    """Serve video from the current online batch."""
    if ONLINE_CURRENT_BATCH is None:
        return 'No active batch', 404
    pairs = _load_online_batch(ONLINE_CURRENT_BATCH)
    if pair_idx < 0 or pair_idx >= len(pairs):
        return 'Not found', 404
    path = pairs[pair_idx]['seg_a'] if segment == 'a' else pairs[pair_idx]['seg_b']
    return send_file(path, mimetype='video/mp4')


@app.route('/api/online/crash_info/<int:pair_idx>')
def api_online_crash_info(pair_idx):
    """Check if segments in this pair contain a crash (-100 reward step)."""
    if ONLINE_CURRENT_BATCH is None:
        return jsonify({'a_crashed': False, 'b_crashed': False})
    pairs = _load_online_batch(ONLINE_CURRENT_BATCH)
    if pair_idx < 0 or pair_idx >= len(pairs):
        return jsonify({'a_crashed': False, 'b_crashed': False})
    meta = pairs[pair_idx]['meta']
    pi = pairs[pair_idx]['pair_idx']
    a_crashed = False
    b_crashed = False
    if meta.get('r_t_1') is not None:
        r1 = meta['r_t_1'][pi]
        a_crashed = bool(np.any(r1 <= -100))
    if meta.get('r_t_2') is not None:
        r2 = meta['r_t_2'][pi]
        b_crashed = bool(np.any(r2 <= -100))
    return jsonify({'a_crashed': a_crashed, 'b_crashed': b_crashed})


## Online mode HTML with back-to-back video playback
ONLINE_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>PEBBLE Online Labeling</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #0f172a; color: #e2e8f0; min-height: 100vh; }
.header { text-align: center; padding: 20px; background: #1e293b; }
.header h1 { font-size: 22px; color: #f8fafc; }
.progress { font-size: 14px; color: #94a3b8; margin-top: 6px; }
.main { max-width: 700px; margin: 20px auto; padding: 0 20px; }
.video-container { text-align: center; margin: 16px 0; }
.segment-label { font-size: 20px; font-weight: 700; padding: 10px; border-radius: 8px;
                  margin-bottom: 10px; display: inline-block; min-width: 200px; }
.seg-a { background: #1e3a5f; color: #60a5fa; }
.seg-b { background: #3b1f3b; color: #c084fc; }
video { width: 100%; max-width: 600px; border-radius: 8px; background: #000; }
.controls { display: flex; gap: 12px; justify-content: center; margin: 20px 0; flex-wrap: wrap; }
.btn { padding: 14px 32px; border: 2px solid transparent; border-radius: 10px;
       font-size: 16px; font-weight: 600; cursor: pointer; transition: all 0.15s; }
.btn:hover { transform: translateY(-2px); }
.btn:disabled { opacity: 0.3; cursor: not-allowed; transform: none; }
.btn-a { background: #1e3a5f; color: #60a5fa; border-color: #2563eb; }
.btn-a:hover:not(:disabled) { background: #2563eb; color: #fff; }
.btn-b { background: #3b1f3b; color: #c084fc; border-color: #7c3aed; }
.btn-b:hover:not(:disabled) { background: #7c3aed; color: #fff; }
.btn-eq { background: #1e293b; color: #94a3b8; border-color: #475569; }
.btn-eq:hover:not(:disabled) { background: #475569; color: #fff; }
.btn-skip { background: #1e293b; color: #64748b; border-color: #334155; }
.btn-skip:hover:not(:disabled) { background: #334155; color: #fff; }
.btn-replay { background: #0f172a; color: #38bdf8; border-color: #0284c7; padding: 10px 20px; font-size: 14px; }
.btn-replay:hover { background: #0284c7; color: #fff; }
.replay-row { display: flex; gap: 12px; justify-content: center; margin: 12px 0; }
.feedback { text-align: center; font-size: 16px; margin: 10px 0; min-height: 24px; color: #22c55e; }
.timer { text-align: center; font-size: 14px; color: #64748b; margin: 6px 0; }
.keyboard-hint { text-align: center; font-size: 13px; color: #475569; margin-top: 8px; }
.waiting { text-align: center; padding: 60px 20px; }
.waiting h2 { font-size: 24px; color: #fbbf24; margin-bottom: 12px; }
.waiting .spinner { font-size: 40px; animation: spin 2s linear infinite; display: inline-block; }
@keyframes spin { 100% { transform: rotate(360deg); } }
.stats { text-align: center; font-size: 13px; color: #475569; margin-top: 20px;
         padding-top: 16px; border-top: 1px solid #1e293b; }
</style>
</head>
<body>
<div class="header">
  <h1>PEBBLE Online Preference Labeling</h1>
  <div class="progress" id="progress">Connecting...</div>
</div>
<div class="main">
  <div id="labeling-ui" style="display:none">
    <div class="video-container">
      <div class="segment-label seg-a" id="seg-indicator">Segment A</div>
      <br>
      <video id="vid" muted></video>
    </div>
    <div class="replay-row">
      <button class="btn btn-replay" id="btn-play" onclick="startPlayback()">&#9654; Play (space)</button>
      <button class="btn btn-replay" onclick="replayBoth()">Replay Both (r)</button>
    </div>
    <div class="timer" id="timer"></div>
    <div class="controls">
      <button class="btn btn-a" id="btn-a" onclick="choose(0)" disabled>A is better (1)</button>
      <button class="btn btn-eq" id="btn-eq" onclick="choose(-1)" disabled>Equal (0)</button>
      <button class="btn btn-b" id="btn-b" onclick="choose(1)" disabled>B is better (2)</button>
      <button class="btn btn-skip" id="btn-skip" onclick="choose(null)" disabled>Skip (s)</button>
    </div>
    <div class="keyboard-hint">Keyboard: SPACE=Play, 1=A, 2=B, 0=Equal, s=Skip, r=Replay</div>
    <div class="feedback" id="feedback"></div>
  </div>
  <div id="waiting-ui" class="waiting">
    <div class="spinner">&#9676;</div>
    <h2>Waiting for training to generate queries...</h2>
    <p id="waiting-msg" style="color:#64748b">The agent is training. When it reaches a query point, segments will appear here.</p>
  </div>
  <div class="stats" id="stats"></div>
</div>
<script>
let pairIdx = 0, totalPairs = 0, batchName = '', labeledTotal = 0;
let startTime = null, currentSeg = 'none', buttonsEnabled = false;
// Track which batch we've already signaled "first Play" for, to measure
// wait-for-Play (server-side: arrival timestamp - first-Play timestamp).
let playSignaledFor = '';

async function pollStatus() {
  try {
    const r = await fetch('/api/online/status');
    const d = await r.json();
    labeledTotal = d.labeled_total;
    document.getElementById('stats').textContent =
      'Total labeled: ' + labeledTotal + '  |  Queue: ' + d.queue_size + ' batches';

    if (d.has_batch) {
      batchName = d.batch_name;
      totalPairs = d.total_pairs;
      pairIdx = d.current_idx;
      document.getElementById('labeling-ui').style.display = '';
      document.getElementById('waiting-ui').style.display = 'none';
      if (pairIdx < totalPairs) {
        updateProgress();
        loadPair(pairIdx);
      }
      return;
    }
  } catch(e) {}

  // No batch available, keep polling
  document.getElementById('labeling-ui').style.display = 'none';
  document.getElementById('waiting-ui').style.display = '';
  setTimeout(pollStatus, 2000);
}

function updateProgress() {
  document.getElementById('progress').textContent =
    batchName + '  |  Pair ' + (pairIdx+1) + '/' + totalPairs +
    '  |  Total labeled: ' + labeledTotal;
}

function setButtons(enabled) {
  buttonsEnabled = enabled;
  document.getElementById('btn-a').disabled = !enabled;
  document.getElementById('btn-b').disabled = !enabled;
  document.getElementById('btn-eq').disabled = !enabled;
  document.getElementById('btn-skip').disabled = !enabled;
}

let crashA = false, crashB = false;
let playbackStarted = false;

function loadPair(i) {
  setButtons(false);
  crashA = false; crashB = false;
  playbackStarted = false;
  const vid = document.getElementById('vid');
  const indicator = document.getElementById('seg-indicator');
  const playBtn = document.getElementById('btn-play');

  // Fetch crash info for this pair
  fetch('/api/online/crash_info/' + i).then(r=>r.json()).then(d => {
    crashA = d.a_crashed;
    crashB = d.b_crashed;
  }).catch(()=>{});

  // Pre-load segment A but don't play yet — wait for user to press Play
  currentSeg = 'ready';
  indicator.textContent = '▶ Press Play to start';
  indicator.className = 'segment-label';
  indicator.style.background = '#1e293b';
  indicator.style.color = '#fbbf24';
  vid.src = '/online_video/' + i + '/a?' + Date.now();
  vid.load();
  vid.onloadeddata = () => {};   // do nothing — wait for click
  vid.onended = null;

  // Show play button, reset its label
  if (playBtn) {
    playBtn.disabled = false;
    playBtn.textContent = '▶ Play (space)';
  }

  // Don't start the timer until user clicks Play
  startTime = null;
  document.getElementById('feedback').textContent = '';
  document.getElementById('timer').textContent = '';
  updateProgress();
}

function startPlayback() {
  if (playbackStarted) return;
  playbackStarted = true;
  // First Play of a fresh batch: tell the server so it can compute
  // wait-for-Play (arrival -> first Play).
  if (batchName && playSignaledFor !== batchName) {
    playSignaledFor = batchName;
    fetch('/api/online/started', {method: 'POST'}).catch(() => {});
  }
  const vid = document.getElementById('vid');
  const indicator = document.getElementById('seg-indicator');
  const playBtn = document.getElementById('btn-play');

  if (playBtn) {
    playBtn.disabled = true;
    playBtn.textContent = 'Playing...';
  }

  // Start playing segment A
  currentSeg = 'a';
  if (crashA) {
    indicator.textContent = 'Segment A — CRASHED';
    indicator.className = 'segment-label';
    indicator.style.background = '#7f1d1d';
    indicator.style.color = '#fca5a5';
  } else {
    indicator.textContent = 'Segment A';
    indicator.className = 'segment-label seg-a';
    indicator.style.background = ''; indicator.style.color = '';
  }
  vid.play();
  vid.onended = () => {
    if (currentSeg === 'a') {
      currentSeg = 'b';
      if (crashB) {
        indicator.textContent = 'Segment B — CRASHED';
        indicator.className = 'segment-label';
        indicator.style.background = '#7f1d1d';
        indicator.style.color = '#fca5a5';
      } else {
        indicator.textContent = 'Segment B';
        indicator.className = 'segment-label seg-b';
        indicator.style.background = ''; indicator.style.color = '';
      }
      vid.src = '/online_video/' + (pairIdx) + '/b?' + Date.now();
      vid.load();
      vid.onloadeddata = () => vid.play();
      vid.onended = () => {
        currentSeg = 'done';
        indicator.textContent = 'Choose your preference';
        indicator.className = 'segment-label';
        indicator.style.background = '#1e293b';
        indicator.style.color = '#22c55e';
        setButtons(true);
      };
    }
  };

  // Start the labeling timer NOW (when user actually starts watching)
  startTime = performance.now();
  updateTimer();
}

function updateTimer() {
  if (startTime) {
    const s = (performance.now() - startTime) / 1000;
    document.getElementById('timer').textContent = 'Time: ' + s.toFixed(1) + 's';
    requestAnimationFrame(updateTimer);
  }
}

function replayBoth() {
  loadPair(pairIdx);
}

async function choose(label) {
  if (!buttonsEnabled && label !== null) return;
  const dt = startTime ? (performance.now() - startTime) / 1000 : 0;
  setButtons(false);

  const r = await fetch('/api/online/label', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({pair_index: pairIdx, label: label, time_sec: dt})
  });
  const d = await r.json();

  if (label !== null) {
    const names = {0:'A preferred', 1:'B preferred', '-1':'Equal'};
    document.getElementById('feedback').textContent =
      (names[String(label)] || 'Labeled') + ' (' + dt.toFixed(1) + 's)';
    labeledTotal++;
  } else {
    document.getElementById('feedback').textContent = 'Skipped';
  }

  if (d.batch_done) {
    document.getElementById('feedback').textContent +=
      ' — Batch complete! Labels sent to training.';
    if (d.has_next) {
      pairIdx = 0;
      updateProgress();
      setTimeout(() => loadPair(0), 1500);
    } else {
      // Go back to waiting
      document.getElementById('labeling-ui').style.display = 'none';
      document.getElementById('waiting-ui').style.display = '';
      document.getElementById('waiting-msg').textContent =
        'Batch done! Waiting for training to generate next query batch...';
      setTimeout(pollStatus, 2000);
    }
  } else {
    pairIdx = d.current_idx;
    updateProgress();
    setTimeout(() => loadPair(pairIdx), 500);
  }
}

document.addEventListener('keydown', e => {
  if (e.key === ' ' || e.code === 'Space') {
    e.preventDefault();
    if (!playbackStarted) startPlayback();
  }
  else if (e.key === '1') choose(0);
  else if (e.key === '2') choose(1);
  else if (e.key === '0') choose(-1);
  else if (e.key === 's' || e.key === 'S') choose(null);
  else if (e.key === 'r' || e.key === 'R') replayBoth();
});

pollStatus();
</script>
</body>
</html>"""


@app.route('/online')
def online_index():
    return ONLINE_HTML


def main():
    global ALL_PAIRS, QUERY_DIR, OUTPUT_PATH, CSV_PATH, ONLINE_MODE, ONLINE_QUERY_DIR

    parser = argparse.ArgumentParser(description="Web-based PEBBLE preference labeling")
    parser.add_argument('--query_dir', type=str, default=None,
                        help='Path to query_videos/ directory (offline mode)')
    parser.add_argument('--mode', choices=['offline', 'online'], default='offline',
                        help='offline: label saved videos, online: watch for new batches from training')
    parser.add_argument('--online_query_dir', type=str, default=None,
                        help='Path to online_queries/ directory (online mode)')
    parser.add_argument('--output', type=str, default='human_labels.pkl',
                        help='Output pickle path')
    parser.add_argument('--csv', type=str, default=None,
                        help='Output CSV path')
    parser.add_argument('--port', type=int, default=8080,
                        help='Web server port')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to bind to')
    args = parser.parse_args()

    OUTPUT_PATH = args.output
    CSV_PATH = args.csv

    if args.mode == 'online':
        if not args.online_query_dir:
            parser.error("--online_query_dir is required in online mode")
        ONLINE_MODE = True
        ONLINE_QUERY_DIR = args.online_query_dir
        QUERY_DIR = args.online_query_dir
        os.makedirs(ONLINE_QUERY_DIR, exist_ok=True)
        print(f"ONLINE MODE — watching: {ONLINE_QUERY_DIR}")
        print(f"Labels saved to: {args.output}")
        print(f"Open in browser: http://localhost:{args.port}/online")
        app.run(host=args.host, port=args.port, debug=False)
    else:
        if not args.query_dir:
            parser.error("--query_dir is required in offline mode")
        QUERY_DIR = args.query_dir
        ALL_PAIRS = load_all_pairs(QUERY_DIR)

    if not ALL_PAIRS:
        print(f"No video pairs found in {args.query_dir}")
        sys.exit(1)

    print(f"Loaded {len(ALL_PAIRS)} video pairs from {args.query_dir}")
    print(f"Labels will be saved to: {args.output}")
    print(f"Open in browser: http://localhost:{args.port}")
    print(f"  (or use SSH tunnel: ssh -L {args.port}:localhost:{args.port} <cluster>)")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == '__main__':
    main()
