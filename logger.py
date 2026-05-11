from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
import json
import os
import csv
import shutil
import torch
import numpy as np
from termcolor import colored

COMMON_TRAIN_FORMAT = [
    ('episode', 'E', 'int'),
    ('step', 'S', 'int'),
    ('episode_reward', 'R', 'float'),
    ('true_episode_reward', 'TR', 'float'), 
    ('total_feedback', 'TF', 'int'),
    ('labeled_feedback', 'LR', 'int'),
    ('noisy_feedback', 'NR', 'int'),
    ('duration', 'D', 'time'),
    ('total_duration', 'TD', 'time'),
]

COMMON_EVAL_FORMAT = [
    ('episode', 'E', 'int'),
    ('step', 'S', 'int'),
    ('episode_reward', 'R', 'float'),
    ('true_episode_reward', 'TR', 'float'),
    ('true_episode_success', 'TS', 'float'),
]


AGENT_TRAIN_FORMAT = {
    'sac': [
        ('batch_reward', 'BR', 'float'),
        ('actor_loss', 'ALOSS', 'float'),
        ('critic_loss', 'CLOSS', 'float'),
        ('alpha_loss', 'TLOSS', 'float'),
        ('alpha_value', 'TVAL', 'float'),
        ('actor_entropy', 'AENT', 'float'),
        ('bc_loss', 'BCLOSS', 'float'),
        # SAC-internal scalars logged by agent.update; listed here so the CSV
        # schema is identical across fresh-write (unified) and append-into-
        # fresh-dir (compare/resume) runs.
        ('actor_target_entropy', 'ATENT', 'float'),
        ('critic_entropy', 'CENT', 'float'),
        ('critic_entropy_max', 'CENTMAX', 'float'),
        ('critic_entropy_min', 'CENTMIN', 'float'),
        ('critic_norm_entropy', 'CNENT', 'float'),
        ('critic_norm_entropy_max', 'CNENTMAX', 'float'),
        ('critic_norm_entropy_min', 'CNENTMIN', 'float'),
    ],
    'ppo': [
        ('batch_reward', 'BR', 'float'),
    ],
}


class AverageMeter(object):
    def __init__(self):
        self._sum = 0
        self._count = 0

    def update(self, value, n=1):
        self._sum += value
        self._count += n

    def value(self):
        return self._sum / max(1, self._count)


class MetersGroup(object):
    def __init__(self, file_name, formating, append=False):
        self._csv_file_name = self._prepare_file(file_name, 'csv', append=append)
        self._formating = formating
        self._meters = defaultdict(AverageMeter)
        self._append = append
        # When appending: skip writing a new header AND lock fieldnames to the
        # existing header. Otherwise the first dump on resume (which has only
        # 'duration' logged) would lock fieldnames to a tiny subset and reject
        # all later rows. Also truncate any partial last line written by a
        # killed previous process.
        self._existing_fieldnames = None
        self._header_written = False
        if append and os.path.exists(self._csv_file_name) \
                and os.path.getsize(self._csv_file_name) > 0:
            self._truncate_partial_last_line(self._csv_file_name)
            with open(self._csv_file_name, 'r', newline='') as f:
                reader = csv.reader(f)
                try:
                    header = next(reader)
                    self._existing_fieldnames = header
                    self._header_written = True
                except StopIteration:
                    pass
        elif append:
            # Resume into a fresh work_dir: no train.csv yet, but we still need
            # to lock the full schema up front so the first dump (which has
            # only 'duration' populated) doesn't pin DictWriter to a tiny
            # subset and crash on the next dump. Derive from the format spec
            # plus 'step' which dump() always adds. Sorted for determinism.
            fieldnames = {k for (k, _, _) in formating}
            fieldnames.add('step')
            self._existing_fieldnames = sorted(fieldnames)
        self._csv_file = open(self._csv_file_name, 'a' if append else 'w')
        self._csv_writer = None

    @staticmethod
    def _truncate_partial_last_line(path):
        # If the last line lacks a trailing newline, the previous process died
        # mid-write; drop it so DictWriter starts on a clean line.
        with open(path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            if size == 0:
                return
            f.seek(size - 1)
            if f.read(1) == b'\n':
                return
        with open(path, 'rb') as f:
            data = f.read()
        last_nl = data.rfind(b'\n')
        if last_nl == -1:
            return  # whole file is one partial line; leave it
        with open(path, 'wb') as f:
            f.write(data[:last_nl + 1])

    def _prepare_file(self, prefix, suffix, append=False):
        file_name = f'{prefix}.{suffix}'
        if (not append) and os.path.exists(file_name):
            os.remove(file_name)
        return file_name

    def log(self, key, value, n=1):
        self._meters[key].update(value, n)

    def _prime_meters(self):
        data = dict()
        for key, meter in self._meters.items():
            if key.startswith('train'):
                key = key[len('train') + 1:]
            else:
                key = key[len('eval') + 1:]
            key = key.replace('/', '_')
            data[key] = meter.value()
        return data

    def _dump_to_csv(self, data):
        if self._csv_writer is None:
            if self._existing_fieldnames is not None:
                # Append mode: reuse existing header. extrasaction='ignore' lets
                # us tolerate a transient first dump that's missing some fields.
                fieldnames = self._existing_fieldnames
                extras = 'ignore'
            else:
                fieldnames = sorted(data.keys())
                extras = 'raise'
            self._csv_writer = csv.DictWriter(self._csv_file,
                                              fieldnames=fieldnames,
                                              restval=0.0,
                                              extrasaction=extras)
            if not self._header_written:
                self._csv_writer.writeheader()
                self._header_written = True
        self._csv_writer.writerow(data)
        self._csv_file.flush()

    def _format(self, key, value, ty):
        if ty == 'int':
            value = int(value)
            return f'{key}: {value}'
        elif ty == 'float':
            return f'{key}: {value:.04f}'
        elif ty == 'time':
            return f'{key}: {value:04.1f} s'
        else:
            raise f'invalid format type: {ty}'

    def _dump_to_console(self, data, prefix):
        prefix = colored(prefix, 'yellow' if prefix == 'train' else 'green')
        pieces = [f'| {prefix: <14}']
        for key, disp_key, ty in self._formating:
            value = data.get(key, 0)
            pieces.append(self._format(disp_key, value, ty))
        print(' | '.join(pieces))

    def dump(self, step, prefix, save=True):
        if len(self._meters) == 0:
            return
        if save:
            data = self._prime_meters()
            data['step'] = step
            self._dump_to_csv(data)
            self._dump_to_console(data, prefix)
        self._meters.clear()


class Logger(object):
    def __init__(self,
                 log_dir,
                 save_tb=False,
                 log_frequency=10000,
                 agent='sac',
                 append=False):
        self._log_dir = log_dir
        self._log_frequency = log_frequency
        if save_tb:
            tb_dir = os.path.join(log_dir, 'tb')
            if (not append) and os.path.exists(tb_dir):
                try:
                    shutil.rmtree(tb_dir)
                except:
                    print("logger.py warning: Unable to remove tb directory")
                    pass
            self._sw = SummaryWriter(tb_dir)
        else:
            self._sw = None
        # each agent has specific output format for training
        assert agent in AGENT_TRAIN_FORMAT
        train_format = COMMON_TRAIN_FORMAT + AGENT_TRAIN_FORMAT[agent]
        self._train_mg = MetersGroup(os.path.join(log_dir, 'train'),
                                     formating=train_format, append=append)
        self._eval_mg = MetersGroup(os.path.join(log_dir, 'eval'),
                                    formating=COMMON_EVAL_FORMAT, append=append)

    def _should_log(self, step, log_frequency):
        log_frequency = log_frequency or self._log_frequency
        return step % log_frequency == 0

    def _try_sw_log(self, key, value, step):
        if self._sw is not None:
            self._sw.add_scalar(key, value, step)

    def _try_sw_log_video(self, key, frames, step):
        if self._sw is not None:
            frames = torch.from_numpy(np.array(frames))
            frames = frames.unsqueeze(0)
            self._sw.add_video(key, frames, step, fps=30)

#    def _try_sw_log_histogram(self, key, histogram, step):
#        if self._sw is not None:
#            self._sw.add_histogram(key, histogram, step)
#
    def _try_sw_log_histogram(self, key, histogram, step):
        try:
            self._sw.add_histogram(key, histogram, step)
        except Exception:
            return

    def log(self, key, value, step, n=1, log_frequency=1):
        if not self._should_log(step, log_frequency):
            return
        assert key.startswith('train') or key.startswith('eval')
        if type(value) == torch.Tensor:
            value = value.item()
        self._try_sw_log(key, value / n, step)
        mg = self._train_mg if key.startswith('train') else self._eval_mg
        mg.log(key, value, n)

    def log_param(self, key, param, step, log_frequency=None):
        if not self._should_log(step, log_frequency):
            return
        self.log_histogram(key + '_w', param.weight.data, step)
        if hasattr(param.weight, 'grad') and param.weight.grad is not None:
            self.log_histogram(key + '_w_g', param.weight.grad.data, step)
        if hasattr(param, 'bias') and hasattr(param.bias, 'data'):
            self.log_histogram(key + '_b', param.bias.data, step)
            if hasattr(param.bias, 'grad') and param.bias.grad is not None:
                self.log_histogram(key + '_b_g', param.bias.grad.data, step)

    def log_video(self, key, frames, step, log_frequency=None):
        if not self._should_log(step, log_frequency):
            return
        assert key.startswith('train') or key.startswith('eval')
        self._try_sw_log_video(key, frames, step)

    def log_histogram(self, key, histogram, step, log_frequency=None):
        if not self._should_log(step, log_frequency):
            return
        assert key.startswith('train') or key.startswith('eval')
        self._try_sw_log_histogram(key, histogram, step)

    def dump(self, step, save=True, ty=None):
        if ty is None:
            self._train_mg.dump(step, 'train', save)
            self._eval_mg.dump(step, 'eval', save)
        elif ty == 'eval':
            self._eval_mg.dump(step, 'eval', save)
        elif ty == 'train':
            self._train_mg.dump(step, 'train', save)
        else:
            raise f'invalid log type: {ty}'
