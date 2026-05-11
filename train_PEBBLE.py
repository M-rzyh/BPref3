#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.expanduser("~/compare_utils"))
from compare_logger import algo_dir, get_clock, Timer
from compute_timer import ComputeTimer

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
#import os
#import sys
import time
import pickle as pkl
import tqdm

from logger import Logger
from replay_buffer import ReplayBuffer
from reward_model import RewardModel
from collections import deque

import gym
import imageio
import subprocess
import utils
import hydra

class Workspace(object):
    def __init__(self, cfg):
        # self.work_dir = os.getcwd()
        # print(f'workspace: {self.work_dir}')
        
        print("\n" + "="*60)
        print("ACTUAL CONFIG BEING USED:")
        print("="*60)
        print(f"env: {cfg.env}")
        print(f"device: {cfg.device}")
        print(f"num_train_steps: {cfg.num_train_steps}")
        print(f"num_unsup_steps: {cfg.num_unsup_steps}")
        print(f"num_interact: {cfg.num_interact}")
        print(f"max_feedback: {cfg.max_feedback}")
        print(f"reward_batch: {cfg.reward_batch}")
        print(f"reward_update: {cfg.reward_update}")
        print(f"feed_type: {cfg.feed_type}")
        print("="*60 + "\n")
        
        # Resume support: if cfg.resume_from is set, derive work_dir from the
        # checkpoint's parent dir so logs (TB, csvs) continue in the same place.
        self.cfg = cfg
        resume_from = getattr(cfg, 'resume_from', None)
        self._resume_from = resume_from if resume_from else None
        if self._resume_from:
            self.work_dir = os.path.dirname(os.path.abspath(self._resume_from))
            print(f"[resume] resuming from {self._resume_from}")
            print(f"[resume] work_dir = {self.work_dir}")
            os.environ.setdefault('COMPARE_RUN_DIR', os.path.dirname(self.work_dir))
        else:
            # Put PEBBLE logs under $COMPARE_RUN_DIR/pebble
            self.work_dir = algo_dir("pebble")
            print(f'workspace: {self.work_dir}')

        self.logger = Logger(
            self.work_dir,
            save_tb=cfg.log_save_tb,
            log_frequency=cfg.log_frequency,
            agent=cfg.agent.name,
            append=bool(self._resume_from))

        self.clock = get_clock()  # shared TB at $COMPARE_RUN_DIR/common_tb
        
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.log_success = False
        
        # make env
        if 'metaworld' in cfg.env:
            self.env = utils.make_metaworld_env(cfg)
            self.log_success = True
        else:
            self.env = utils.make_env(cfg)

        self.max_episode_steps = utils.get_env_horizon(self.env)
        
        cfg.agent.params.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.params.action_dim = self.env.action_space.shape[0]
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent = hydra.utils.instantiate(cfg.agent)

        self.replay_buffer = ReplayBuffer(
            self.env.observation_space.shape,
            self.env.action_space.shape,
            int(cfg.replay_buffer_capacity),
            self.device)
        
        # for logging
        self.total_feedback = 0
        self.labeled_feedback = 0
        self.step = 0
        self.last_pref_sec = 0.0
        self.last_pref_pairs = 0
        self.pref_time_so_far_sec = 0.0
        self.last_logged_labeled_feedback = 0

        # query video saving
        self.query_batch_idx = 0
        self._query_video_dir = None
        self._query_index = []  # list of batch paths for index.pkl

        # online web labeling (feed_type=7)
        self._online_query_dir = None

        # checkpoint-at-max-feedback flags
        self._should_checkpoint = False
        self._checkpoint_saved = False  # write-once guard
        self._checkpoint_done = False   # loop-exit signal (only set if cfg says exit)

        # per-batch timing breakdown (one dict per learn_reward call)
        self._batch_timings = []
        self._last_batch_end_t = None  # set when run() starts

        # cumulative human/pref labeling time (actual measured, not extrapolated)
        self.human_pref_sec = 0.0
        # Web-only: time the labeling UI was open but human hadn't pressed Play
        # yet (sum across all web batches).
        self.cumulative_wait_for_start_sec = 0.0

        # instantiating the reward model
        self.reward_model = RewardModel(
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            ensemble_size=cfg.ensemble_size,
            size_segment=cfg.segment,
            activation=cfg.activation, 
            lr=cfg.reward_lr,
            mb_size=cfg.reward_batch, 
            large_batch=cfg.large_batch, 
            label_margin=cfg.label_margin, 
            teacher_beta=cfg.teacher_beta, 
            teacher_gamma=cfg.teacher_gamma, 
            teacher_eps_mistake=cfg.teacher_eps_mistake, 
            teacher_eps_skip=cfg.teacher_eps_skip, 
            teacher_eps_equal=cfg.teacher_eps_equal)

        # Start a persistent Xvfb for video recording
        self._xvfb_proc = None
        # Xvfb is needed whenever something will render frames:
        #  - save_video (post-query eval videos)
        #  - save_query_videos (offline labeling videos)
        #  - feed_type=7 (web mode renders segment videos for live UI)
        need_xvfb = (cfg.save_video
                     or getattr(cfg, 'save_query_videos', False)
                     or int(cfg.feed_type) == 7)
        if need_xvfb and cfg.env.startswith('gym_'):
            display_num = 99 + os.getpid() % 900
            self._xvfb_proc = subprocess.Popen(
                ['Xvfb', f':{display_num}', '-screen', '0', '1024x768x24', '-nolisten', 'tcp'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            os.environ['DISPLAY'] = f':{display_num}'
            time.sleep(0.5)
            print(f"[video] Xvfb started on :{display_num}")

        # Resume: load checkpoint after all components (agent/replay/reward) are built
        if self._resume_from:
            self.load_checkpoint(self._resume_from)

    def evaluate(self):
        average_episode_reward = 0
        average_true_episode_reward = 0
        success_rate = 0
        
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            self.agent.reset()
            done = False
            episode_reward = 0
            true_episode_reward = 0
            if self.log_success:
                episode_success = 0

            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, extra = self.env.step(action)
                
                episode_reward += reward
                true_episode_reward += reward
                if self.log_success:
                    episode_success = max(episode_success, extra['success'])
                
            average_episode_reward += episode_reward
            average_true_episode_reward += true_episode_reward
            if self.log_success:
                success_rate += episode_success
            
        average_episode_reward /= self.cfg.num_eval_episodes
        average_true_episode_reward /= self.cfg.num_eval_episodes
        if self.log_success:
            success_rate /= self.cfg.num_eval_episodes
            success_rate *= 100.0
        
        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step)
        self.logger.log('eval/true_episode_reward', average_true_episode_reward,
                        self.step)
        if self.log_success:
            self.logger.log('eval/success_rate', success_rate,
                    self.step)
            self.logger.log('train/true_episode_success', success_rate,
                        self.step)
        self.logger.dump(self.step)

    def record_video(self):
        """Record one episode of the current policy and save as .mp4."""
        if not self.cfg.save_video:
            return
        if not self.cfg.env.startswith('gym_'):
            return

        job_id = os.environ.get('SLURM_JOB_ID', 'local')
        video_dir = os.path.join(self.work_dir, 'videos', job_id)
        os.makedirs(video_dir, exist_ok=True)
        video_path = os.path.join(video_dir, f'step_{self.step:08d}.mp4')

        try:
            env_id = self.cfg.env[len('gym_'):]
            rec_env = gym.make(env_id)

            # Hard cap so an early hovering policy can't render forever
            max_record_steps = int(getattr(self.cfg, 'max_episode_steps', None) or 400)

            frames = []
            obs = rec_env.reset()
            done = False
            ep_reward = 0.0
            step_i = 0
            while not done and step_i < max_record_steps:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, _ = rec_env.step(action)
                ep_reward += reward
                frame = rec_env.render(mode='rgb_array')
                frames.append(frame)
                step_i += 1
            rec_env.close()

            if frames:
                imageio.mimwrite(video_path, frames)
                print(f"[video] saved {video_path}  ({len(frames)} frames, reward={ep_reward:.1f})")
        except Exception as e:
            print(f"[video] failed to record: {e}")

    def record_final_videos(self, num_episodes=5):
        """Record multiple full episodes after training is done."""
        if not self.cfg.env.startswith('gym_'):
            return

        video_dir = os.path.join(self.work_dir, 'videos')
        os.makedirs(video_dir, exist_ok=True)

        env_id = self.cfg.env[len('gym_'):]
        max_record_steps = int(getattr(self.cfg, 'max_episode_steps', None) or 400)
        for ep in range(num_episodes):
            video_path = os.path.join(video_dir, f'final_ep{ep}.mp4')
            try:
                rec_env = gym.make(env_id)
                frames = []
                obs = rec_env.reset()
                done = False
                ep_reward = 0.0
                step_i = 0
                while not done and step_i < max_record_steps:
                    with utils.eval_mode(self.agent):
                        action = self.agent.act(obs, sample=False)
                    obs, reward, done, _ = rec_env.step(action)
                    ep_reward += reward
                    frame = rec_env.render(mode='rgb_array')
                    step_i += 1
                    frames.append(frame)
                rec_env.close()

                if frames:
                    imageio.mimwrite(video_path, frames)
                    print(f"[video] final ep{ep}: {video_path}  ({len(frames)} frames, reward={ep_reward:.1f})")
            except Exception as e:
                print(f"[video] final ep{ep} failed: {e}")

    def _save_query_videos(self):
        """Render and save segment videos for the last query batch."""
        lp = self.reward_model._last_put
        if lp is None or lp['sa_t_1'].shape[0] == 0:
            return

        # Determine save directory
        if self._query_video_dir is None:
            compare_dir = os.environ.get('COMPARE_RUN_DIR', '')
            if compare_dir:
                self._query_video_dir = os.path.join(
                    compare_dir, 'query_videos')
            else:
                self._query_video_dir = os.path.join(
                    self.work_dir, 'query_videos')
                print("[warn] COMPARE_RUN_DIR not set, saving query videos "
                      f"to {self._query_video_dir}")
            os.makedirs(self._query_video_dir, exist_ok=True)

        # Create a temporary env for rendering
        if self.cfg.env.startswith('gym_'):
            env_id = self.cfg.env[len('gym_'):]
        else:
            print("[save_query_videos] skipping — only gym envs supported")
            return

        try:
            render_env = gym.make(env_id)
            render_env.reset()

            # We need r_t for metadata — re-derive from reward model targets
            # or pass None since oracle labels are already in _last_put
            self.reward_model.save_query_batch(
                sa_t_1=lp['sa_t_1'],
                sa_t_2=lp['sa_t_2'],
                r_t_1=None,
                r_t_2=None,
                len_1=lp['len_1'],
                len_2=lp['len_2'],
                oracle_labels=lp['labels'],
                batch_idx=self.query_batch_idx,
                save_dir=self._query_video_dir,
                env=render_env,
            )

            batch_path = os.path.join(
                self._query_video_dir, f'batch_{self.query_batch_idx:03d}')
            self._query_index.append(batch_path)
            self.query_batch_idx += 1

            render_env.close()
        except Exception as e:
            print(f"[save_query_videos] failed: {e}")

    def _web_sampling(self):
        """Online web-based human labeling via shared filesystem.

        If cfg.web_replay_from is set, first try to read a saved (segments +
        labels) bundle from that directory. If found, apply directly without
        rendering or waiting. Otherwise falls through to live web_sampling.
        """
        # ---- Replay path (recovery from a crashed run) ----
        replay_dir = getattr(self.cfg, 'web_replay_from', None)
        if replay_dir:
            count = self.reward_model.web_sampling_replay(replay_dir)
            if count >= 0:
                return count
            # count == -1 means no saved batch at this idx → fall through

        if self._online_query_dir is None:
            compare_dir = os.environ.get('COMPARE_RUN_DIR', '')
            if compare_dir:
                self._online_query_dir = os.path.join(
                    compare_dir, 'online_queries')
            else:
                self._online_query_dir = os.path.join(
                    self.work_dir, 'online_queries')
            os.makedirs(self._online_query_dir, exist_ok=True)
            print(f"[web_sampling] query dir: {self._online_query_dir}")

        if not self.cfg.env.startswith('gym_'):
            raise RuntimeError("web_sampling only supports gym envs")

        env_id = self.cfg.env[len('gym_'):]
        render_env = gym.make(env_id)
        render_env.reset()
        try:
            count = self.reward_model.web_sampling(
                self._online_query_dir, render_env,
                poll_interval=getattr(self.cfg, 'web_poll_interval', 2.0),
                timeout=getattr(self.cfg, 'web_timeout', 3600))
        finally:
            render_env.close()
        return count

    def _switch_phase(self, new_phase):
        """Accumulate time spent in current phase, then switch."""
        now = time.perf_counter()
        self._phase_sec[self._phase_current] += now - self._phase_last_t
        self._phase_last_t = now
        self._phase_current = new_phase

    # ------------------------------------------------------------------
    # Checkpoint save / load (full training-state snapshot)
    # ------------------------------------------------------------------

    def _get_env_np_random_state(self):
        """Snapshot env.np_random (gym's internal RNG). Returns None if env
        doesn't expose it (e.g. metaworld wrappers)."""
        try:
            rng = getattr(self.env.unwrapped, 'np_random', None)
            return rng.get_state() if rng is not None else None
        except Exception:
            return None

    def _set_env_np_random_state(self, state):
        if state is None:
            return
        try:
            rng = getattr(self.env.unwrapped, 'np_random', None)
            if rng is not None:
                rng.set_state(state)
        except Exception as e:
            print(f"[checkpoint] could not restore env.np_random: {e}")

    def save_checkpoint(self, path):
        """Save EVERY piece of stateful info needed to resume training.

        Memory-efficient + atomic:
        - The big replay buffer arrays go to a sibling `.replay.npz` (numpy
          zero-copy save, no pickle, no 2x RAM spike).
        - Everything else (small tensors, optimizer states, counters, RNG)
          goes to `.pt`.
        - Both files are written to `.tmp` then atomically renamed.
        """
        import random as _random
        import json as _json

        rm = self.reward_model
        rb = self.replay_buffer

        # ---- Replay buffer to .npz (memory-friendly) ----
        # np.savez auto-appends '.npz' if path doesn't end in it, so tmp must
        # already end in '.npz' or the subsequent rename will look for a
        # non-existent file.
        replay_path = path + '.replay.npz'
        replay_tmp = path + '.replay.tmp.npz'
        np.savez(
            replay_tmp,
            obses=rb.obses,
            next_obses=rb.next_obses,
            actions=rb.actions,
            rewards=rb.rewards,
            not_dones=rb.not_dones,
            not_dones_no_max=rb.not_dones_no_max,
            idx=np.array(rb.idx),
            full=np.array(rb.full),
        )
        os.rename(replay_tmp, replay_path)

        ckpt = {
            'version': 2,
            'replay_npz_path': os.path.basename(replay_path),  # sibling file
            # ---- SAC agent ----
            'sac': {
                'actor': self.agent.actor.state_dict(),
                'critic': self.agent.critic.state_dict(),
                'critic_target': self.agent.critic_target.state_dict(),
                'log_alpha': self.agent.log_alpha.detach().cpu(),
                'actor_optim': self.agent.actor_optimizer.state_dict(),
                'critic_optim': self.agent.critic_optimizer.state_dict(),
                'log_alpha_optim': self.agent.log_alpha_optimizer.state_dict(),
            },
            # ---- Reward model ----
            'reward': {
                'ensemble': [m.state_dict() for m in rm.ensemble],
                'optim': rm.opt.state_dict(),
                'pref_buffer': {
                    'seg1': rm.buffer_seg1, 'seg2': rm.buffer_seg2,
                    'label': rm.buffer_label,
                    'len1': rm.buffer_len1, 'len2': rm.buffer_len2,
                    'index': rm.buffer_index, 'full': rm.buffer_full,
                },
                'traj': {
                    'inputs': rm.inputs, 'targets': rm.targets,
                    'raw_actions': rm.raw_actions,
                },
                'misc': {
                    'mb_size': rm.mb_size,
                    'origin_mb_size': rm.origin_mb_size,
                    'teacher_thres_skip': rm.teacher_thres_skip,
                    'teacher_thres_equal': rm.teacher_thres_equal,
                    'online_batch_idx': rm._online_batch_idx,
                    'last_web_human_time': rm._last_web_human_time,
                    'last_web_wait_sec': getattr(rm, '_last_web_wait_sec', 0.0),
                },
            },
            # ---- Replay buffer is in the sibling .replay.npz (see top) ----
            # ---- Workspace counters ----
            'workspace': {
                'step': self.step,
                'total_feedback': self.total_feedback,
                'labeled_feedback': self.labeled_feedback,
                'human_pref_sec': self.human_pref_sec,
                'cumulative_wait_for_start_sec': self.cumulative_wait_for_start_sec,
                'pref_time_so_far_sec': self.pref_time_so_far_sec,
                'last_pref_sec': self.last_pref_sec,
                'last_pref_pairs': self.last_pref_pairs,
                'last_logged_labeled_feedback': self.last_logged_labeled_feedback,
                'phase_sec': dict(self._phase_sec),
                'query_batch_idx': self.query_batch_idx,
                'query_index': list(self._query_index),
                'human_labels_loaded': getattr(self, '_human_labels_loaded', False),
                'avg_train_true_return': list(self._avg_train_true_return),
                'avg_episode_length': list(self._avg_episode_length),
                'interact_count': self._interact_count,
            },
            # ---- RNG state for reproducibility ----
            'rng': {
                'numpy': np.random.get_state(),
                'torch': torch.get_rng_state(),
                'torch_cuda': (torch.cuda.get_rng_state_all()
                               if torch.cuda.is_available() else None),
                'python': _random.getstate(),
                # env.np_random is the env's INTERNAL RNG (separate from numpy
                # global). Saving it lets the resume's first env.reset()
                # produce the same fresh obs as the unified's first env.reset()
                # after the checkpoint.
                'env_np_random': self._get_env_np_random_state(),
            },
            # ---- Config snapshot for sanity check on resume ----
            'cfg_env': str(self.cfg.env),
            'cfg_seed': int(self.cfg.seed),
            'cfg_num_train_steps': int(self.cfg.num_train_steps),
            'cfg_max_feedback': int(self.cfg.max_feedback),
        }

        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        # Atomic write: tmp + rename so a kill mid-write never leaves a half file
        tmp_path = path + '.tmp'
        torch.save(ckpt, tmp_path)
        os.rename(tmp_path, path)

        # Also write a JSON sidecar listing what's inside (for debugging)
        meta_path = path + '.meta.json'
        meta = {
            'path': path,
            'version': 1,
            'step': int(self.step),
            'total_feedback': int(self.total_feedback),
            'env': str(self.cfg.env),
            'seed': int(self.cfg.seed),
            'work_dir': self.work_dir,
            'sections': sorted(ckpt.keys()),
            'pref_buffer_index': int(rm.buffer_index),
            'pref_buffer_full': bool(rm.buffer_full),
            'replay_idx': int(rb.idx),
            'replay_full': bool(rb.full),
            'replay_size': int(rb.capacity if rb.full else rb.idx),
        }
        with open(meta_path, 'w') as f:
            _json.dump(meta, f, indent=2)
        print(f"[checkpoint] saved -> {path}")
        print(f"[checkpoint] meta  -> {meta_path}")
        print(f"[checkpoint] step={self.step}  total_feedback={self.total_feedback}  "
              f"replay_size={meta['replay_size']}")

    def load_checkpoint(self, path):
        """Restore everything saved by save_checkpoint(). Must be called AFTER
        Workspace.__init__ has constructed agent/reward_model/replay_buffer."""
        import random as _random

        ckpt = torch.load(path, map_location=self.device)
        if ckpt.get('version', 0) != 1:
            print(f"[checkpoint] WARNING: unknown version {ckpt.get('version')}")

        # Sanity: env / seed match
        if ckpt.get('cfg_env') != str(self.cfg.env):
            print(f"[checkpoint] WARNING: env mismatch "
                  f"(ckpt={ckpt.get('cfg_env')}, cfg={self.cfg.env})")

        # ---- SAC ----
        s = ckpt['sac']
        self.agent.actor.load_state_dict(s['actor'])
        self.agent.critic.load_state_dict(s['critic'])
        self.agent.critic_target.load_state_dict(s['critic_target'])
        # log_alpha is a tensor with requires_grad — restore value, keep grad
        with torch.no_grad():
            self.agent.log_alpha.copy_(s['log_alpha'].to(self.device))
        self.agent.actor_optimizer.load_state_dict(s['actor_optim'])
        self.agent.critic_optimizer.load_state_dict(s['critic_optim'])
        self.agent.log_alpha_optimizer.load_state_dict(s['log_alpha_optim'])

        # ---- Reward model ----
        r = ckpt['reward']
        for m, sd in zip(self.reward_model.ensemble, r['ensemble']):
            m.load_state_dict(sd)
        self.reward_model.opt.load_state_dict(r['optim'])
        pb = r['pref_buffer']
        self.reward_model.buffer_seg1 = pb['seg1']
        self.reward_model.buffer_seg2 = pb['seg2']
        self.reward_model.buffer_label = pb['label']
        self.reward_model.buffer_len1 = pb['len1']
        self.reward_model.buffer_len2 = pb['len2']
        self.reward_model.buffer_index = pb['index']
        self.reward_model.buffer_full = pb['full']
        tj = r['traj']
        self.reward_model.inputs = tj['inputs']
        self.reward_model.targets = tj['targets']
        self.reward_model.raw_actions = tj['raw_actions']
        ms = r['misc']
        self.reward_model.mb_size = ms['mb_size']
        self.reward_model.origin_mb_size = ms.get('origin_mb_size', ms['mb_size'])
        self.reward_model.teacher_thres_skip = ms['teacher_thres_skip']
        self.reward_model.teacher_thres_equal = ms['teacher_thres_equal']
        self.reward_model._online_batch_idx = ms['online_batch_idx']
        self.reward_model._last_web_human_time = ms.get('last_web_human_time', 0.0)
        self.reward_model._last_web_wait_sec = ms.get('last_web_wait_sec', 0.0)

        # ---- Replay buffer ----
        if 'replay' in ckpt:
            # v1 format: replay was inside the .pt file
            rb_data = ckpt['replay']
            self.replay_buffer.obses = rb_data['obses']
            self.replay_buffer.next_obses = rb_data['next_obses']
            self.replay_buffer.actions = rb_data['actions']
            self.replay_buffer.rewards = rb_data['rewards']
            self.replay_buffer.not_dones = rb_data['not_dones']
            self.replay_buffer.not_dones_no_max = rb_data['not_dones_no_max']
            self.replay_buffer.idx = rb_data['idx']
            self.replay_buffer.full = rb_data['full']
        else:
            # v2 format: replay buffer in sibling .npz file
            replay_basename = ckpt.get('replay_npz_path',
                                       os.path.basename(path) + '.replay.npz')
            replay_path = os.path.join(os.path.dirname(path), replay_basename)
            with np.load(replay_path) as data:
                self.replay_buffer.obses = data['obses']
                self.replay_buffer.next_obses = data['next_obses']
                self.replay_buffer.actions = data['actions']
                self.replay_buffer.rewards = data['rewards']
                self.replay_buffer.not_dones = data['not_dones']
                self.replay_buffer.not_dones_no_max = data['not_dones_no_max']
                self.replay_buffer.idx = int(data['idx'])
                self.replay_buffer.full = bool(data['full'])

        # ---- Workspace counters ----
        w = ckpt['workspace']
        self.step = w['step']
        self.total_feedback = w['total_feedback']
        self.labeled_feedback = w['labeled_feedback']
        # Backward-compat: old checkpoints saved this as 'cumulative_pref_sec'
        self.human_pref_sec = w.get('human_pref_sec',
                                     w.get('cumulative_pref_sec', 0.0))
        self.cumulative_wait_for_start_sec = w.get('cumulative_wait_for_start_sec', 0.0)
        self.pref_time_so_far_sec = w['pref_time_so_far_sec']
        self.last_pref_sec = w['last_pref_sec']
        self.last_pref_pairs = w['last_pref_pairs']
        self.last_logged_labeled_feedback = w['last_logged_labeled_feedback']
        # Phase timing accumulators are restored so the resumed run continues counting
        self._phase_sec_resumed = dict(w['phase_sec'])
        self.query_batch_idx = w['query_batch_idx']
        self._query_index = list(w['query_index'])
        if w['human_labels_loaded']:
            self._human_labels_loaded = True
        from collections import deque
        self._avg_train_true_return = deque(w['avg_train_true_return'], maxlen=10)
        self._avg_episode_length = deque(w['avg_episode_length'], maxlen=10)
        self._interact_count = w['interact_count']

        # ---- RNG ----
        # torch.load(map_location='cuda') moves saved tensors to GPU; RNG states
        # must be CPU ByteTensors, so coerce explicitly.
        rng = ckpt.get('rng', {})
        if 'numpy' in rng:
            np.random.set_state(rng['numpy'])
        if 'torch' in rng:
            torch.set_rng_state(rng['torch'].cpu().to(torch.uint8))
        if rng.get('torch_cuda') is not None and torch.cuda.is_available():
            cuda_states = [s.cpu().to(torch.uint8) for s in rng['torch_cuda']]
            torch.cuda.set_rng_state_all(cuda_states)
        if 'python' in rng:
            _random.setstate(rng['python'])
        if 'env_np_random' in rng:
            self._set_env_np_random_state(rng['env_np_random'])

        print(f"[checkpoint] loaded <- {path}")
        print(f"[checkpoint] resumed at step={self.step}  "
              f"total_feedback={self.total_feedback}  "
              f"replay_size={self.replay_buffer.capacity if self.replay_buffer.full else self.replay_buffer.idx}")

    def _record_batch_timing(self, idx, step_at_query, prev_window_sec,
                             learn_reward_sec, relabel_sec, video_sec,
                             reset_update_sec=0.0,
                             human_time_sec=0.0, wait_for_start_sec=0.0):
        """Append one batch's timing breakdown and print a one-line summary."""
        entry = {
            'batch_idx': idx,
            'step_at_query': int(step_at_query),
            'prev_window_sec': float(prev_window_sec),
            'learn_reward_sec': float(learn_reward_sec),
            'relabel_sec': float(relabel_sec),
            'video_sec': float(video_sec),
            'reset_update_sec': float(reset_update_sec),
            'human_time_sec': float(human_time_sec),
            'wait_for_start_sec': float(wait_for_start_sec),
            'total_overhead_sec': float(learn_reward_sec + relabel_sec
                                        + video_sec + reset_update_sec),
        }
        self._batch_timings.append(entry)
        print(f"[batch {idx}] step={step_at_query}  "
              f"prev_window={prev_window_sec:6.1f}s  "
              f"learn_reward={learn_reward_sec:6.1f}s  "
              f"relabel={relabel_sec:5.1f}s  "
              f"video={video_sec:4.1f}s"
              + (f"  reset_update={reset_update_sec:5.1f}s" if reset_update_sec > 0 else "")
              + (f"  human={human_time_sec:5.1f}s" if human_time_sec > 0 else "")
              + (f"  wait_play={wait_for_start_sec:5.1f}s" if wait_for_start_sec > 0 else ""))

    def _maybe_checkpoint_and_exit(self):
        """If checkpoint flag set, save full state and (optionally) signal main
        loop to break. Save and exit are decoupled via
        cfg.exit_after_max_feedback_checkpoint."""
        if not self._should_checkpoint or self._checkpoint_saved:
            return False
        ckpt_path = os.path.join(self.work_dir, 'checkpoint.pt')
        print(f"\n[checkpoint] max_feedback hit (total={self.total_feedback}). "
              f"Saving full training state...")
        self.save_checkpoint(ckpt_path)
        self._checkpoint_saved = True
        print(f"[checkpoint] To resume the pure-RL portion, submit a job with:")
        print(f"           resume_from={ckpt_path}")
        if getattr(self.cfg, 'exit_after_max_feedback_checkpoint', True):
            self._checkpoint_done = True
            return True
        print("[checkpoint] exit_after_max_feedback_checkpoint=false -> "
              "continuing training to num_train_steps")
        return False

    def learn_reward(self, first_flag=0):
        # --- offline human labels: load once, skip further querying ---
        human_path = getattr(self.cfg, 'human_labels_path', None)
        if human_path and not getattr(self, '_human_labels_loaded', False):
            labeled_queries = self.reward_model.load_human_labels(human_path)
            self._human_labels_loaded = True
            # Load actual human labeling time from the CSV
            csv_path = human_path.replace('.pkl', '.csv')
            if os.path.exists(csv_path):
                import csv as _csv
                with open(csv_path) as f:
                    rows = list(_csv.DictReader(f))
                total_human_sec = sum(
                    float(r['time_sec']) for r in rows
                    if r.get('label') not in (None, 'None', ''))
                self.human_pref_sec = total_human_sec
                print(f"[human_labels] loaded labeling time from CSV: {total_human_sec:.1f}s")
            t_pref_dt = 0.0
        elif human_path and self._human_labels_loaded:
            # Labels already loaded — skip querying, just retrain below
            labeled_queries = 0
            t_pref_dt = 0.0
        else:
            # --- preference query generation + labeling timing ---
            labeled_queries = 0
            with Timer() as  t_pref:
                if first_flag == 1:
                    labeled_queries = self.reward_model.uniform_sampling()
                else:
                    if self.cfg.feed_type == 0:
                        labeled_queries = self.reward_model.uniform_sampling()
                    elif self.cfg.feed_type == 1:
                        labeled_queries = self.reward_model.disagreement_sampling()
                    elif self.cfg.feed_type == 2:
                        labeled_queries = self.reward_model.entropy_sampling()
                    elif self.cfg.feed_type == 3:
                        labeled_queries = self.reward_model.kcenter_sampling()
                    elif self.cfg.feed_type == 4:
                        labeled_queries = self.reward_model.kcenter_disagree_sampling()
                    elif self.cfg.feed_type == 5:
                        labeled_queries = self.reward_model.kcenter_entropy_sampling()
                    elif self.cfg.feed_type == 6:
                        labeled_queries = self.reward_model.human_sampling()
                    elif self.cfg.feed_type == 7:
                        labeled_queries = self._web_sampling()
                    else:
                        raise NotImplementedError
                t_pref_dt = t_pref.dt

        pref_sec = t_pref_dt if t_pref_dt is not None else 0.0

        # Save query videos if enabled (renders on-demand, no frame storage)
        if getattr(self.cfg, 'save_query_videos', False) and self.reward_model._last_put is not None:
            self._save_query_videos()

        # # get feedbacks
        # labeled_queries, noisy_queries = 0, 0
        # if first_flag == 1:
        #     # if it is first time to get feedback, need to use random sampling
        #     labeled_queries = self.reward_model.uniform_sampling()
        # else:
        #     if self.cfg.feed_type == 0:
        #         labeled_queries = self.reward_model.uniform_sampling()
        #     elif self.cfg.feed_type == 1:
        #         labeled_queries = self.reward_model.disagreement_sampling()
        #     elif self.cfg.feed_type == 2:
        #         labeled_queries = self.reward_model.entropy_sampling()
        #     elif self.cfg.feed_type == 3:
        #         labeled_queries = self.reward_model.kcenter_sampling()
        #     elif self.cfg.feed_type == 4:
        #         labeled_queries = self.reward_model.kcenter_disagree_sampling()
        #     elif self.cfg.feed_type == 5:
        #         labeled_queries = self.reward_model.kcenter_entropy_sampling()
        #     else:
        #         raise NotImplementedError
        
        self.total_feedback += labeled_queries
        self.labeled_feedback += labeled_queries
        query_number = self.total_feedback / max(1, self.cfg.reward_batch)
        print(f"Total preference labels so far: {self.total_feedback}, Approx query batches: {query_number:.2f}")

        # Auto-checkpoint trigger: when max_feedback hit and flag is set, mark
        # for clean shutdown after this learn_reward returns.
        if (getattr(self.cfg, 'checkpoint_at_max_feedback', False)
                and self.total_feedback >= self.cfg.max_feedback
                and not getattr(self, '_checkpoint_saved', False)):
            self._should_checkpoint = True
        
        seg_len = int(self.cfg.segment)             # L
        pairs = int(labeled_queries)               # number of preference labels created now
        segment_steps = 2 * seg_len * pairs        # 2*L per preference
        
        self.last_pref_sec = float(pref_sec)
        self.last_pref_pairs = int(pairs)
        self.pref_time_so_far_sec = (self.last_pref_sec / max(1, self.last_pref_pairs)) * self.labeled_feedback

        # Cumulative HUMAN time (the actual decision-making time, not wall-clock).
        # For feed_type=7 (web), web_sampling stores the real human time on the
        # reward model (_last_web_human_time), which excludes polling/render
        # overhead. For other modes, fall back to the wrapping Timer's wall-clock.
        if int(self.cfg.feed_type) == 7 and hasattr(self.reward_model, '_last_web_human_time'):
            self.human_pref_sec += float(self.reward_model._last_web_human_time)
            # Web-only: accumulate "web open, human hasn't pressed Play yet" time
            self.cumulative_wait_for_start_sec += float(
                getattr(self.reward_model, '_last_web_wait_sec', 0.0))
        else:
            self.human_pref_sec += float(pref_sec)
        
        # self.clock.log_scalar("clock/pebble_pref_batch_seconds", pref_sec, self.step)
        # self.clock.log_scalar("samples/pebble_pref_pairs", float(pairs), self.step)
        # self.clock.log_scalar("samples/pebble_segment_len", float(seg_len), self.step)
        # self.clock.log_scalar("samples/pebble_pref_segment_steps", float(segment_steps), self.step)

        # self.clock.log_scalar("true_reward/sample", self.true_episode_reward,(pref_sec / max(1, pairs))*len(self.labeled_feedback))
        # self.clock.log_scalar("reward/sample", self.episode_reward,(pref_sec / max(1, pairs))*len(self.labeled_feedback))

        # self.clock.log_scalar(
        #     "clock/pebble_pref_sec_per_segment_step",
        #     pref_sec / max(1, segment_steps),
        #     self.step
        # )
        self.clock.flush()
        
        train_acc = 0
        total_acc = 0.0
        if self.labeled_feedback > 0:
            # update reward
            for epoch in range(self.cfg.reward_update):
                if self.cfg.label_margin > 0 or self.cfg.teacher_eps_equal > 0 or self.cfg.feed_type in (6, 7):
                    train_acc = self.reward_model.train_soft_reward()
                else:
                    train_acc = self.reward_model.train_reward()
                total_acc = np.mean(train_acc)
                
                if total_acc > 0.97:
                    break;
                    
        print("Reward function is updated!! ACC: " + str(total_acc))

    def run(self):
        # Start compute timer for the whole training run
        self._compute_timer = ComputeTimer()
        self._compute_timer.start()

        # ----- Phase timing (printed to .out at end of run) -----
        # phases:
        #   pretrain      = seed + unsupervised exploration (before first query)
        #   queries       = cumulative time inside learn_reward calls
        #   interactions  = cumulative env-step time between queries (before pure_rl)
        #   pure_rl       = env-step + agent-update time after max_feedback exhausted
        # If resumed from a checkpoint, restore accumulated phase times.
        if hasattr(self, '_phase_sec_resumed'):
            self._phase_sec = dict(self._phase_sec_resumed)
            # Resumed run is in pure_rl (no more queries), so start there.
            self._phase_current = 'pure_rl'
            del self._phase_sec_resumed
        else:
            self._phase_sec = {'pretrain': 0.0, 'queries': 0.0,
                               'interactions': 0.0, 'pure_rl': 0.0}
            self._phase_current = 'pretrain'
        self._phase_last_t = time.perf_counter()
        if self._last_batch_end_t is None:
            self._last_batch_end_t = time.perf_counter()

        episode, episode_reward, done = 0, 0, True
        episode_step = 0
        if self.log_success:
            episode_success = 0
        true_episode_reward = 0

        # store train returns of recent 10 episodes (exposed on self for ckpt)
        if not hasattr(self, '_avg_train_true_return'):
            self._avg_train_true_return = deque([], maxlen=10)
            self._avg_episode_length = deque([], maxlen=10)
        avg_train_true_return = self._avg_train_true_return
        avg_episode_length = self._avg_episode_length
        start_time = time.time()
        env_time_acc = 0.0
        env_steps_acc = 0
        ENV_LOG_EVERY = 1000

        # interact_count: env steps since last query (exposed on self for ckpt)
        if not hasattr(self, '_interact_count'):
            self._interact_count = 0
        interact_count = self._interact_count
        while self.step < self.cfg.num_train_steps:
            # Guard: if checkpoint was just taken, exit cleanly
            if self._checkpoint_done:
                print("[checkpoint] exiting training loop after checkpoint")
                break
            if done:
                if self.step > 0:
                    self.logger.log('train/duration', time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps))

                # evaluate agent periodically
                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()
                
                self.logger.log('train/episode_reward', episode_reward, self.step)
                self.logger.log('train/true_episode_reward', true_episode_reward, self.step)
                print(f"[step {self.step}] episode {episode} finished in {episode_step} steps")
                
                # rough clock timing
                if self.labeled_feedback > self.last_logged_labeled_feedback:
                    print("\nLogging to TB: total feedback", self.total_feedback, "labeled feedback", self.labeled_feedback, "pref time so far (sec)", self.pref_time_so_far_sec)
                    x_ms = int(self.pref_time_so_far_sec * 1000)

                    self.clock.log_scalar("1: true_reward/sample", true_episode_reward, x_ms)
                    self.clock.log_scalar("1: reward/sample", episode_reward, x_ms)
                    self.clock.flush()

                    self.last_logged_labeled_feedback = self.labeled_feedback
                
                self.clock.log_scalar("2: true_reward/sample", true_episode_reward,int(((self.last_pref_sec / max(1, self.last_pref_pairs))*(self.labeled_feedback))))
                self.clock.log_scalar("2: reward/sample", episode_reward,int(((self.last_pref_sec / max(1, self.last_pref_pairs))*(self.labeled_feedback))))

                # Clean cumulative timing: x-axis = actual measured pref time in ms
                if self.human_pref_sec > 0:
                    x_cum_ms = int(self.human_pref_sec * 1000)
                    self.clock.log_scalar("pebble/true_reward_vs_human_time", true_episode_reward, x_cum_ms)
                    self.clock.log_scalar("pebble/reward_vs_human_time", episode_reward, x_cum_ms)
                    self.clock.flush()
                
                self.logger.log('train/total_feedback', self.total_feedback, self.step)
                self.logger.log('train/labeled_feedback', self.labeled_feedback, self.step)
                
                if self.log_success:
                    self.logger.log('train/episode_success', episode_success,
                        self.step)
                    self.logger.log('train/true_episode_success', episode_success,
                        self.step)

                # Episode-boundary checkpoint: triggered when max_feedback was
                # hit during this episode. Saving here (BEFORE env.reset())
                # means the resume's first env.reset() consumes the same
                # env.np_random state -> identical fresh obs in both runs.
                if self._should_checkpoint and not self._checkpoint_saved:
                    if self._maybe_checkpoint_and_exit():
                        break  # exit_after_max_feedback_checkpoint=true

                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                avg_train_true_return.append(true_episode_reward)
                avg_episode_length.append(episode_step)
                true_episode_reward = 0
                if self.log_success:
                    episode_success = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)
                        
            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update                
            if self.step == (self.cfg.num_seed_steps + self.cfg.num_unsup_steps):
                # update schedule
                if self.cfg.reward_schedule == 1:
                    frac = (self.cfg.num_train_steps-self.step) / self.cfg.num_train_steps
                    if frac == 0:
                        frac = 0.01
                elif self.cfg.reward_schedule == 2:
                    frac = self.cfg.num_train_steps / (self.cfg.num_train_steps-self.step +1)
                else:
                    frac = 1
                self.reward_model.change_batch(frac)
                
                # update margin --> use average actual episode length
                horizon = np.mean(avg_episode_length) if len(avg_episode_length) > 0 else max(1, int(self.cfg.segment))
                horizon = max(horizon, 1)
                new_margin = np.mean(avg_train_true_return) * (self.cfg.segment / horizon)
                self.reward_model.set_teacher_thres_skip(new_margin)
                self.reward_model.set_teacher_thres_equal(new_margin)

                # first learn reward — pretrain ends here, queries phase begins
                _batch_idx = self.reward_model._online_batch_idx
                _batch_step = self.step
                _prev_window = time.perf_counter() - self._last_batch_end_t
                self._switch_phase('queries')
                _t0 = time.perf_counter()
                self.learn_reward(first_flag=0)
                _learn_sec = time.perf_counter() - _t0
                self._switch_phase('interactions')
                # NOTE: checkpoint is saved at next episode boundary (see if-done
                # block) so the env state aligns with the resume's fresh
                # env.reset(). Saving mid-episode caused divergence.

                # relabel buffer
                _t0 = time.perf_counter()
                self.replay_buffer.relabel_with_predictor(self.reward_model)
                _relabel_sec = time.perf_counter() - _t0
                _t0 = time.perf_counter()
                self.record_video()
                _video_sec = time.perf_counter() - _t0

                # reset Q due to unsuperivsed exploration
                self.agent.reset_critic()

                # update agent
                _t0 = time.perf_counter()
                self.agent.update_after_reset(
                    self.replay_buffer, self.logger, self.step,
                    gradient_update=self.cfg.reset_update,
                    policy_update=True)
                _reset_sec = time.perf_counter() - _t0

                self._record_batch_timing(_batch_idx, _batch_step, _prev_window,
                                          _learn_sec, _relabel_sec, _video_sec,
                                          reset_update_sec=_reset_sec,
                                          human_time_sec=float(getattr(self.reward_model, '_last_web_human_time', 0.0)),
                                          wait_for_start_sec=float(getattr(self.reward_model, '_last_web_wait_sec', 0.0)))
                self._last_batch_end_t = time.perf_counter()

                # reset interact_count
                interact_count = 0
            elif self.step > self.cfg.num_seed_steps + self.cfg.num_unsup_steps:
                # update reward function
                if self.total_feedback < self.cfg.max_feedback:
                    if interact_count == self.cfg.num_interact:
                        # update schedule
                        if self.cfg.reward_schedule == 1:
                            frac = (self.cfg.num_train_steps-self.step) / self.cfg.num_train_steps
                            if frac == 0:
                                frac = 0.01
                        elif self.cfg.reward_schedule == 2:
                            frac = self.cfg.num_train_steps / (self.cfg.num_train_steps-self.step +1)
                        else:
                            frac = 1
                        self.reward_model.change_batch(frac)
                        
                        # update margin --> use average actual episode length
                        horizon = np.mean(avg_episode_length) if len(avg_episode_length) > 0 else max(1, int(self.cfg.segment))
                        horizon = max(horizon, 1)
                        new_margin = np.mean(avg_train_true_return) * (self.cfg.segment / horizon)
                        self.reward_model.set_teacher_thres_skip(new_margin * self.cfg.teacher_eps_skip)
                        self.reward_model.set_teacher_thres_equal(new_margin * self.cfg.teacher_eps_equal)
                        
                        # corner case: new total feed > max feed
                        if self.reward_model.mb_size + self.total_feedback > self.cfg.max_feedback:
                            self.reward_model.set_batch(self.cfg.max_feedback - self.total_feedback)
                            
                        # subsequent query
                        _batch_idx = self.reward_model._online_batch_idx
                        _batch_step = self.step
                        _prev_window = time.perf_counter() - self._last_batch_end_t
                        self._switch_phase('queries')
                        _t0 = time.perf_counter()
                        self.learn_reward()
                        _learn_sec = time.perf_counter() - _t0
                        # If max_feedback exhausted, transition to pure_rl from now on
                        if self.total_feedback >= self.cfg.max_feedback:
                            self._switch_phase('pure_rl')
                        else:
                            self._switch_phase('interactions')
                        _t0 = time.perf_counter()
                        self.replay_buffer.relabel_with_predictor(self.reward_model)
                        _relabel_sec = time.perf_counter() - _t0
                        _t0 = time.perf_counter()
                        self.record_video()
                        _video_sec = time.perf_counter() - _t0
                        self._record_batch_timing(_batch_idx, _batch_step, _prev_window,
                                                  _learn_sec, _relabel_sec, _video_sec,
                                                  human_time_sec=float(getattr(self.reward_model, '_last_web_human_time', 0.0)),
                                                  wait_for_start_sec=float(getattr(self.reward_model, '_last_web_wait_sec', 0.0)))
                        self._last_batch_end_t = time.perf_counter()
                        interact_count = 0
                        # NOTE: checkpoint defers to next episode boundary; see
                        # if-done block in main loop.

                self.agent.update(self.replay_buffer, self.logger, self.step, 1)
                
            # unsupervised exploration
            elif self.step > self.cfg.num_seed_steps:
                self.agent.update_state_ent(self.replay_buffer, self.logger, self.step, 
                                            gradient_update=1, K=self.cfg.topK)
                
            # next_obs, reward, done, extra = self.env.step(action)
            t0 = time.perf_counter()
            next_obs, reward, done, extra = self.env.step(action)
            dt = time.perf_counter() - t0
            env_time_acc += dt
            env_steps_acc += 1
            if env_steps_acc >= ENV_LOG_EVERY:
                self.clock.log_scalar("clock/env_step_sec", env_time_acc / env_steps_acc, self.step)
                self.clock.flush()
                env_time_acc = 0.0
                env_steps_acc = 0
                
            reward_hat = self.reward_model.r_hat(np.concatenate([obs, action], axis=-1))

            # allow infinite bootstrap
            done = float(done)
            if self.max_episode_steps is None:
                done_no_max = done
            else:
                done_no_max = 0 if episode_step + 1 == self.max_episode_steps else done
            episode_reward += reward_hat
            true_episode_reward += reward
            
            if self.log_success:
                episode_success = max(episode_success, extra['success'])
                
            # adding data to the reward training data
            self.reward_model.add_data(obs, action, reward, done)
            self.replay_buffer.add(
                obs, action, reward_hat, 
                next_obs, done, done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1
            interact_count += 1
            self._interact_count = interact_count

        self.agent.save(self.work_dir, self.step)
        self.reward_model.save(self.work_dir, self.step)
        self.reward_model.save_trajectories(
            os.path.join(self.work_dir, 'trajectories.pkl'))
        self.record_final_videos(num_episodes=5)

        # Save query video index for offline labeling
        if self._query_index:
            index_path = os.path.join(self._query_video_dir, 'index.pkl')
            with open(index_path, 'wb') as f:
                pkl.dump(self._query_index, f)
            print(f"[query_videos] saved index ({len(self._query_index)} "
                  f"batches) -> {index_path}")

        # Flush final phase time accumulator
        self._switch_phase(self._phase_current)  # records remaining time in current phase

        # Print phase timing summary
        p = self._phase_sec
        active_sec = p['pretrain'] + p['queries'] + p['interactions']
        total_sec = active_sec + p['pure_rl']
        human_sec = float(getattr(self, 'human_pref_sec', 0.0))
        wait_play_sec = float(getattr(self, 'cumulative_wait_for_start_sec', 0.0))
        # For web mode, queries phase wall-clock includes the time waiting for
        # the human; subtracting that gives the pure compute portion of queries.
        compute_only_sec = total_sec - human_sec - wait_play_sec
        print("\n" + "=" * 70)
        print("PHASE TIMING SUMMARY  (all values are wall-clock seconds)")
        print("=" * 70)
        print(f"  Pre-training (seed + unsup):     {p['pretrain']:9.1f} sec")
        print(f"  Queries (cumulative)*:           {p['queries']:9.1f} sec")
        print(f"  Interactions between queries:    {p['interactions']:9.1f} sec")
        print(f"  Pure RL (after max_feedback):    {p['pure_rl']:9.1f} sec")
        print("  " + "-" * 60)
        print(f"  Active (pretrain+query+inter):   {active_sec:9.1f} sec")
        print(f"  TOTAL wall-clock training:       {total_sec:9.1f} sec")
        print()
        print(f"  Human labeling (decision time):  {human_sec:9.1f} sec  <- Play -> click, real human work")
        print(f"  Web idle (wait for Play):        {wait_play_sec:9.1f} sec  <- batch arrived, human hadn't started")
        print(f"  Compute-only (total - human - wait): {compute_only_sec:9.1f} sec  <- env steps + RL + reward training")
        print()
        print("  * Queries phase wall-clock INCLUDES waiting for human labels (web/inline modes).")
        print("    'Human labeling' is the actual decision time reported by the labeler.")
        print("    'TOTAL wall-clock training' already includes human and wait time.")
        print("=" * 70)

        # Per-batch breakdown
        if self._batch_timings:
            print("\nPER-BATCH TIMING BREAKDOWN")
            print("-" * 110)
            print(f"  {'batch':<5} {'step':>8} {'prev_window':>12} {'learn_reward':>14}"
                  f" {'relabel':>9} {'video':>7} {'reset_upd':>10}"
                  f" {'human':>8} {'wait_play':>10}")
            for b in self._batch_timings:
                print(f"  {b['batch_idx']:<5} {b['step_at_query']:>8} "
                      f"{b['prev_window_sec']:>11.1f}s {b['learn_reward_sec']:>13.1f}s "
                      f"{b['relabel_sec']:>8.1f}s {b['video_sec']:>6.1f}s "
                      f"{b['reset_update_sec']:>9.1f}s "
                      f"{b.get('human_time_sec', 0.0):>7.1f}s "
                      f"{b.get('wait_for_start_sec', 0.0):>9.1f}s")
            print("-" * 110)
            print("  prev_window = wall-clock between end of previous batch and this batch's query")
            print("                (= pretrain phase for batch 0; ~20K env steps for later batches)")
            print("  human       = Play->click decision time for this batch")
            print("  wait_play   = time batch was open in browser before human pressed Play")
            print()

        # Stop compute timer and log / save stats
        compute_stats = self._compute_timer.stop()
        extra = {
            'algo': 'pebble',
            'env': self.cfg.env,
            'feed_type': int(self.cfg.feed_type),
            'num_train_steps': int(self.cfg.num_train_steps),
            'seed': int(self.cfg.seed),
            'human_pref_sec': float(getattr(self, 'human_pref_sec', 0.0)),
            'cumulative_wait_for_start_sec': float(getattr(self, 'cumulative_wait_for_start_sec', 0.0)),
            'phase_pretrain_sec': float(p['pretrain']),
            'phase_queries_sec': float(p['queries']),
            'phase_interactions_sec': float(p['interactions']),
            'phase_pure_rl_sec': float(p['pure_rl']),
            'phase_active_sec': float(active_sec),
            'phase_total_sec': float(total_sec),
            'phase_compute_only_sec': float(compute_only_sec),
        }
        compute_path = os.path.join(self.work_dir, 'compute_time.json')
        self._compute_timer.save_json(compute_path, compute_stats, extra=extra)
        print(f"[compute] wall={compute_stats['wall_sec']:.1f}s "
              f"cpu={compute_stats['cpu_sec']:.1f}s "
              f"gpu={compute_stats['gpu_sec']:.1f}s "
              f"parallel={compute_stats['parallelism_factor']:.2f}x "
              f"-> {compute_path}")
        # Also log to TB (at final step)
        try:
            self.clock.log_scalar('compute/wall_sec', compute_stats['wall_sec'], self.step)
            self.clock.log_scalar('compute/cpu_sec', compute_stats['cpu_sec'], self.step)
            self.clock.log_scalar('compute/gpu_sec', compute_stats['gpu_sec'], self.step)
            self.clock.log_scalar('compute/parallelism_factor',
                                  compute_stats['parallelism_factor'], self.step)
            self.clock.flush()
        except Exception as e:
            print(f"[compute] TB logging failed: {e}")

@hydra.main(config_path='config/train_PEBBLE.yaml', strict=True)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()

if __name__ == '__main__':
    main()
