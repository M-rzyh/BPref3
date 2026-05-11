import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import itertools
import tqdm
import copy
import scipy.stats as st
import os
import time

from scipy.stats import norm

import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def gen_net(in_size=1, out_size=1, H=128, n_layers=3, activation='tanh'):
    net = []
    for i in range(n_layers):
        net.append(nn.Linear(in_size, H))
        net.append(nn.LeakyReLU())
        in_size = H
    net.append(nn.Linear(in_size, out_size))
    if activation == 'tanh':
        net.append(nn.Tanh())
    elif activation == 'sig':
        net.append(nn.Sigmoid())
    else:
        net.append(nn.ReLU())

    return net

def KCenterGreedy(obs, full_obs, num_new_sample):
    selected_index = []
    current_index = list(range(obs.shape[0]))
    new_obs = obs
    new_full_obs = full_obs
    start_time = time.time()
    for count in range(num_new_sample):
        dist = compute_smallest_dist(new_obs, new_full_obs)
        max_index = torch.argmax(dist)
        max_index = max_index.item()
        
        if count == 0:
            selected_index.append(max_index)
        else:
            selected_index.append(current_index[max_index])
        current_index = current_index[0:max_index] + current_index[max_index+1:]
        
        new_obs = obs[current_index]
        new_full_obs = np.concatenate([
            full_obs, 
            obs[selected_index]], 
            axis=0)
    return selected_index

def compute_smallest_dist(obs, full_obs):
    obs = torch.from_numpy(obs).float()
    full_obs = torch.from_numpy(full_obs).float()
    batch_size = 100
    with torch.no_grad():
        total_dists = []
        for full_idx in range(len(obs) // batch_size + 1):
            full_start = full_idx * batch_size
            if full_start < len(obs):
                full_end = (full_idx + 1) * batch_size
                dists = []
                for idx in range(len(full_obs) // batch_size + 1):
                    start = idx * batch_size
                    if start < len(full_obs):
                        end = (idx + 1) * batch_size
                        dist = torch.norm(
                            obs[full_start:full_end, None, :].to(device) - full_obs[None, start:end, :].to(device), dim=-1, p=2
                        )
                        dists.append(dist)
                dists = torch.cat(dists, dim=1)
                small_dists = torch.torch.min(dists, dim=1).values
                total_dists.append(small_dists)
                
        total_dists = torch.cat(total_dists)
    return total_dists.unsqueeze(1)

class RewardModel:
    def __init__(self, ds, da, 
                 ensemble_size=3, lr=3e-4, mb_size = 128, size_segment=1, 
                 env_maker=None, max_size=100, activation='tanh', capacity=5e5,  
                 large_batch=1, label_margin=0.0, 
                 teacher_beta=-1, teacher_gamma=1, 
                 teacher_eps_mistake=0, 
                 teacher_eps_skip=0, 
                 teacher_eps_equal=0):
        
        # train data is trajectories, must process to sa and s..   
        self.ds = ds
        self.da = da
        self.de = ensemble_size
        self.lr = lr
        self.ensemble = []
        self.paramlst = []
        self.opt = None
        self.model = None
        self.max_size = max_size
        self.activation = activation
        self.size_segment = size_segment
        
        self.capacity = int(capacity)
        self.buffer_seg1 = np.empty((self.capacity, size_segment, self.ds+self.da), dtype=np.float32)
        self.buffer_seg2 = np.empty((self.capacity, size_segment, self.ds+self.da), dtype=np.float32)
        self.buffer_label = np.empty((self.capacity, 1), dtype=np.float32)
        self.buffer_len1 = np.full((self.capacity,), size_segment, dtype=np.int32)
        self.buffer_len2 = np.full((self.capacity,), size_segment, dtype=np.int32)
        self.buffer_index = 0
        self.buffer_full = False
                
        self.construct_ensemble()
        self.inputs = []
        self.targets = []
        self.raw_actions = []
        self.img_inputs = []
        self.mb_size = mb_size
        self.origin_mb_size = mb_size
        self.train_batch_size = 128
        self.CEloss = nn.CrossEntropyLoss()
        self.running_means = []
        self.running_stds = []
        self.best_seg = []
        self.best_label = []
        self.best_action = []
        self.large_batch = large_batch

        # Last stored query batch (for video saving)
        self._last_put = None

        # Online web labeling state
        self._online_batch_idx = 0
        self._last_web_human_time = 0.0
        # Time the labeling UI was waiting before the human first pressed Play
        # (set on each web batch from response['wait_for_start_sec']).
        self._last_web_wait_sec = 0.0

        # new teacher
        self.teacher_beta = teacher_beta
        self.teacher_gamma = teacher_gamma
        self.teacher_eps_mistake = teacher_eps_mistake
        self.teacher_eps_equal = teacher_eps_equal
        self.teacher_eps_skip = teacher_eps_skip
        self.teacher_thres_skip = 0
        self.teacher_thres_equal = 0
        
        self.label_margin = label_margin
        self.label_target = 1 - 2*self.label_margin
    
    def softXEnt_loss(self, input, target):
        logprobs = torch.nn.functional.log_softmax (input, dim = 1)
        return  -(target * logprobs).sum() / input.shape[0]
    
    def change_batch(self, new_frac):
        self.mb_size = int(self.origin_mb_size*new_frac)
    
    def set_batch(self, new_batch):
        self.mb_size = int(new_batch)
        
    def set_teacher_thres_skip(self, new_margin):
        self.teacher_thres_skip = new_margin * self.teacher_eps_skip
        
    def set_teacher_thres_equal(self, new_margin):
        self.teacher_thres_equal = new_margin * self.teacher_eps_equal
        
    def construct_ensemble(self):
        for i in range(self.de):
            model = nn.Sequential(*gen_net(in_size=self.ds+self.da, 
                                           out_size=1, H=256, n_layers=3, 
                                           activation=self.activation)).float().to(device)
            self.ensemble.append(model)
            self.paramlst.extend(model.parameters())
            
        self.opt = torch.optim.Adam(self.paramlst, lr = self.lr)
            
    def add_data(self, obs, act, rew, done):
        sa_t = np.concatenate([obs, act], axis=-1)
        r_t = rew
        
        flat_input = sa_t.reshape(1, self.da+self.ds)
        r_t = np.array(r_t)
        flat_target = r_t.reshape(1, 1)

        init_data = len(self.inputs) == 0
        if init_data:
            self.inputs.append(flat_input)
            self.targets.append(flat_target)
        elif done:
            self.inputs[-1] = np.concatenate([self.inputs[-1], flat_input])
            self.targets[-1] = np.concatenate([self.targets[-1], flat_target])
            # FIFO
            if len(self.inputs) > self.max_size:
                self.inputs = self.inputs[1:]
                self.targets = self.targets[1:]
            self.inputs.append([])
            self.targets.append([])
        else:
            if len(self.inputs[-1]) == 0:
                self.inputs[-1] = flat_input
                self.targets[-1] = flat_target
            else:
                self.inputs[-1] = np.concatenate([self.inputs[-1], flat_input])
                self.targets[-1] = np.concatenate([self.targets[-1], flat_target])
                
    def add_data_batch(self, obses, rewards):
        num_env = obses.shape[0]
        for index in range(num_env):
            self.inputs.append(obses[index])
            self.targets.append(rewards[index])
        
    def get_rank_probability(self, x_1, x_2, len_1, len_2):
        # get probability x_1 > x_2
        probs = []
        for member in range(self.de):
            probs.append(self.p_hat_member(x_1, x_2, len_1, len_2, member=member).cpu().numpy())
        probs = np.array(probs)

        return np.mean(probs, axis=0), np.std(probs, axis=0)

    def get_entropy(self, x_1, x_2, len_1, len_2):
        # get probability x_1 > x_2
        probs = []
        for member in range(self.de):
            probs.append(self.p_hat_entropy(x_1, x_2, len_1, len_2, member=member).cpu().numpy())
        probs = np.array(probs)
        return np.mean(probs, axis=0), np.std(probs, axis=0)

    def p_hat_member(self, x_1, x_2, len_1, len_2, member=-1):
        # softmaxing to get the probabilities according to eqn 1
        with torch.no_grad():
            r_hat1 = self.r_hat_member(x_1, member=member)
            r_hat2 = self.r_hat_member(x_2, member=member)
            len_1_t = torch.from_numpy(len_1).long().to(device)
            len_2_t = torch.from_numpy(len_2).long().to(device)
            r_hat1 = self._masked_sum_torch(r_hat1, len_1_t)
            r_hat2 = self._masked_sum_torch(r_hat2, len_2_t)
            r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

        # taking 0 index for probability x_1 > x_2
        return F.softmax(r_hat, dim=-1)[:,0]

    def p_hat_entropy(self, x_1, x_2, len_1, len_2, member=-1):
        # softmaxing to get the probabilities according to eqn 1
        with torch.no_grad():
            r_hat1 = self.r_hat_member(x_1, member=member)
            r_hat2 = self.r_hat_member(x_2, member=member)
            len_1_t = torch.from_numpy(len_1).long().to(device)
            len_2_t = torch.from_numpy(len_2).long().to(device)
            r_hat1 = self._masked_sum_torch(r_hat1, len_1_t)
            r_hat2 = self._masked_sum_torch(r_hat2, len_2_t)
            r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

        ent = F.softmax(r_hat, dim=-1) * F.log_softmax(r_hat, dim=-1)
        ent = ent.sum(axis=-1).abs()
        return ent

    def r_hat_member(self, x, member=-1):
        # the network parameterizes r hat in eqn 1 from the paper
        return self.ensemble[member](torch.from_numpy(x).float().to(device))

    def r_hat(self, x):
        # they say they average the rewards from each member of the ensemble, but I think this only makes sense if the rewards are already normalized
        # but I don't understand how the normalization should be happening right now :(
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member(x, member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)
        return np.mean(r_hats)
    
    def r_hat_batch(self, x):
        # they say they average the rewards from each member of the ensemble, but I think this only makes sense if the rewards are already normalized
        # but I don't understand how the normalization should be happening right now :(
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member(x, member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)

        return np.mean(r_hats, axis=0)

    def _masked_sum_torch(self, values, lengths):
        """Sum values (batch, seg_len, 1) only over the first lengths[i] steps."""
        seg_len = values.shape[1]
        mask = torch.arange(seg_len, device=values.device).unsqueeze(0) < lengths.unsqueeze(1)
        return (values.squeeze(-1) * mask.float()).sum(dim=1, keepdim=True)

    def _masked_sum_np(self, values, lengths):
        """Sum values (batch, seg_len, 1) only over the first lengths[i] steps."""
        seg_len = values.shape[1]
        mask = np.arange(seg_len)[None, :] < lengths[:, None]
        return (values.squeeze(-1) * mask).sum(axis=1, keepdims=True)

    def save(self, model_dir, step):
        for member in range(self.de):
            torch.save(
                self.ensemble[member].state_dict(), '%s/reward_model_%s_%s.pt' % (model_dir, step, member)
            )
            
    def load(self, model_dir, step):
        for member in range(self.de):
            self.ensemble[member].load_state_dict(
                torch.load('%s/reward_model_%s_%s.pt' % (model_dir, step, member))
            )
    
    def get_train_acc(self):
        ensemble_acc = np.array([0 for _ in range(self.de)])
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = np.random.permutation(max_len)
        batch_size = 256
        num_epochs = int(np.ceil(max_len/batch_size))

        total = 0
        for epoch in range(num_epochs):
            last_index = (epoch+1)*batch_size
            if (epoch+1)*batch_size > max_len:
                last_index = max_len

            sa_t_1 = self.buffer_seg1[epoch*batch_size:last_index]
            sa_t_2 = self.buffer_seg2[epoch*batch_size:last_index]
            labels_np = self.buffer_label[epoch*batch_size:last_index].flatten()
            hard_labels = torch.from_numpy(np.round(labels_np).astype(int)).long().to(device)
            len_1_t = torch.from_numpy(self.buffer_len1[epoch*batch_size:last_index]).long().to(device)
            len_2_t = torch.from_numpy(self.buffer_len2[epoch*batch_size:last_index]).long().to(device)
            total += len(labels_np)
            for member in range(self.de):
                # get logits
                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)
                r_hat1 = self._masked_sum_torch(r_hat1, len_1_t)
                r_hat2 = self._masked_sum_torch(r_hat2, len_2_t)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == hard_labels).sum().item()
                ensemble_acc[member] += correct

        ensemble_acc = ensemble_acc / total
        return np.mean(ensemble_acc)
    
    def _pad_to_segment(self, arr):
        """Zero-pad arr (L, D) to (size_segment, D)."""
        pad_len = self.size_segment - len(arr)
        if pad_len <= 0:
            return arr
        pad = np.zeros((pad_len, arr.shape[1]), dtype=arr.dtype)
        return np.concatenate([arr, pad], axis=0)

    def get_queries(self, mb_size=20):
        if mb_size <= 0 or len(self.inputs) == 0:
            return self._empty_query_batch()

        eligible_inputs = []
        eligible_targets = []
        for idx, (traj_input, traj_target) in enumerate(zip(self.inputs, self.targets)):
            if isinstance(traj_input, list) or isinstance(traj_target, list):
                continue

            # The last trajectory can still be in-progress if no terminal transition was observed yet.
            if idx == len(self.inputs) - 1 and not isinstance(self.inputs[-1], list):
                continue

            if len(traj_input) == 0:
                continue

            eligible_inputs.append(traj_input)
            eligible_targets.append(traj_target)

        if len(eligible_inputs) == 0:
            return self._empty_query_batch()

        batch_index_1 = np.random.choice(len(eligible_inputs), size=mb_size, replace=True)
        batch_index_2 = np.random.choice(len(eligible_inputs), size=mb_size, replace=True)

        sa_t_1_list, sa_t_2_list = [], []
        r_t_1_list, r_t_2_list = [], []
        len_1_list, len_2_list = [], []

        for idx_1, idx_2 in zip(batch_index_1, batch_index_2):
            traj_sa_1 = eligible_inputs[idx_1]
            traj_r_1 = eligible_targets[idx_1]
            traj_sa_2 = eligible_inputs[idx_2]
            traj_r_2 = eligible_targets[idx_2]

            # Extract segment from trajectory 1
            if len(traj_sa_1) >= self.size_segment:
                start_1 = np.random.randint(0, len(traj_sa_1) - self.size_segment + 1)
                seg_sa_1 = traj_sa_1[start_1:start_1 + self.size_segment]
                seg_r_1 = traj_r_1[start_1:start_1 + self.size_segment]
                l1 = self.size_segment
            else:
                seg_sa_1 = self._pad_to_segment(traj_sa_1)
                seg_r_1 = self._pad_to_segment(traj_r_1)
                l1 = len(traj_sa_1)

            # Extract segment from trajectory 2
            if len(traj_sa_2) >= self.size_segment:
                start_2 = np.random.randint(0, len(traj_sa_2) - self.size_segment + 1)
                seg_sa_2 = traj_sa_2[start_2:start_2 + self.size_segment]
                seg_r_2 = traj_r_2[start_2:start_2 + self.size_segment]
                l2 = self.size_segment
            else:
                seg_sa_2 = self._pad_to_segment(traj_sa_2)
                seg_r_2 = self._pad_to_segment(traj_r_2)
                l2 = len(traj_sa_2)

            sa_t_1_list.append(seg_sa_1)
            r_t_1_list.append(seg_r_1)
            sa_t_2_list.append(seg_sa_2)
            r_t_2_list.append(seg_r_2)
            len_1_list.append(l1)
            len_2_list.append(l2)

        sa_t_1 = np.asarray(sa_t_1_list, dtype=np.float32)
        sa_t_2 = np.asarray(sa_t_2_list, dtype=np.float32)
        r_t_1 = np.asarray(r_t_1_list, dtype=np.float32)
        r_t_2 = np.asarray(r_t_2_list, dtype=np.float32)
        len_1 = np.array(len_1_list, dtype=np.int32)
        len_2 = np.array(len_2_list, dtype=np.int32)

        return sa_t_1, sa_t_2, r_t_1, r_t_2, len_1, len_2

    def _empty_query_batch(self):
        empty_sa = np.empty((0, self.size_segment, self.ds + self.da), dtype=np.float32)
        empty_r = np.empty((0, self.size_segment, 1), dtype=np.float32)
        empty_len = np.empty((0,), dtype=np.int32)
        return empty_sa, empty_sa.copy(), empty_r, empty_r.copy(), empty_len, empty_len.copy()

    def put_queries(self, sa_t_1, sa_t_2, labels, len_1, len_2):
        # Capture last stored batch for optional video saving
        self._last_put = {
            'sa_t_1': sa_t_1.copy(),
            'sa_t_2': sa_t_2.copy(),
            'labels': labels.copy() if hasattr(labels, 'copy') else np.array(labels),
            'len_1': len_1.copy(),
            'len_2': len_2.copy(),
        }
        total_sample = sa_t_1.shape[0]
        next_index = self.buffer_index + total_sample
        if next_index >= self.capacity:
            self.buffer_full = True
            maximum_index = self.capacity - self.buffer_index
            np.copyto(self.buffer_seg1[self.buffer_index:self.capacity], sa_t_1[:maximum_index])
            np.copyto(self.buffer_seg2[self.buffer_index:self.capacity], sa_t_2[:maximum_index])
            np.copyto(self.buffer_label[self.buffer_index:self.capacity], labels[:maximum_index])
            np.copyto(self.buffer_len1[self.buffer_index:self.capacity], len_1[:maximum_index])
            np.copyto(self.buffer_len2[self.buffer_index:self.capacity], len_2[:maximum_index])

            remain = total_sample - (maximum_index)
            if remain > 0:
                np.copyto(self.buffer_seg1[0:remain], sa_t_1[maximum_index:])
                np.copyto(self.buffer_seg2[0:remain], sa_t_2[maximum_index:])
                np.copyto(self.buffer_label[0:remain], labels[maximum_index:])
                np.copyto(self.buffer_len1[0:remain], len_1[maximum_index:])
                np.copyto(self.buffer_len2[0:remain], len_2[maximum_index:])

            self.buffer_index = remain
        else:
            np.copyto(self.buffer_seg1[self.buffer_index:next_index], sa_t_1)
            np.copyto(self.buffer_seg2[self.buffer_index:next_index], sa_t_2)
            np.copyto(self.buffer_label[self.buffer_index:next_index], labels)
            np.copyto(self.buffer_len1[self.buffer_index:next_index], len_1)
            np.copyto(self.buffer_len2[self.buffer_index:next_index], len_2)
            self.buffer_index = next_index
            
    def get_label(self, sa_t_1, sa_t_2, r_t_1, r_t_2, len_1, len_2):
        sum_r_t_1 = self._masked_sum_np(r_t_1, len_1)
        sum_r_t_2 = self._masked_sum_np(r_t_2, len_2)

        # skip the query
        if self.teacher_thres_skip > 0:
            max_r_t = np.maximum(sum_r_t_1, sum_r_t_2)
            max_index = (max_r_t > self.teacher_thres_skip).reshape(-1)
            if sum(max_index) == 0:
                return None, None, None, None, None, None, []

            sa_t_1 = sa_t_1[max_index]
            sa_t_2 = sa_t_2[max_index]
            r_t_1 = r_t_1[max_index]
            r_t_2 = r_t_2[max_index]
            len_1 = len_1[max_index]
            len_2 = len_2[max_index]
            sum_r_t_1 = self._masked_sum_np(r_t_1, len_1)
            sum_r_t_2 = self._masked_sum_np(r_t_2, len_2)

        # equally preferable
        margin_index = (np.abs(sum_r_t_1 - sum_r_t_2) < self.teacher_thres_equal).reshape(-1)

        # perfectly rational — apply teacher_gamma discount within actual lengths
        seg_size = r_t_1.shape[1]
        temp_r_t_1 = r_t_1.copy()
        temp_r_t_2 = r_t_2.copy()
        for index in range(seg_size-1):
            temp_r_t_1[:,:index+1] *= self.teacher_gamma
            temp_r_t_2[:,:index+1] *= self.teacher_gamma
        sum_r_t_1 = self._masked_sum_np(temp_r_t_1, len_1)
        sum_r_t_2 = self._masked_sum_np(temp_r_t_2, len_2)

        rational_labels = 1*(sum_r_t_1 < sum_r_t_2)
        if self.teacher_beta > 0: # Bradley-Terry rational model
            r_hat = torch.cat([torch.Tensor(sum_r_t_1),
                               torch.Tensor(sum_r_t_2)], axis=-1)
            r_hat = r_hat*self.teacher_beta
            ent = F.softmax(r_hat, dim=-1)[:, 1]
            labels = torch.bernoulli(ent).int().numpy().reshape(-1, 1)
        else:
            labels = rational_labels

        # making a mistake
        len_labels = labels.shape[0]
        rand_num = np.random.rand(len_labels)
        noise_index = rand_num <= self.teacher_eps_mistake
        labels[noise_index] = 1 - labels[noise_index]

        # equally preferable
        labels[margin_index] = -1

        return sa_t_1, sa_t_2, r_t_1, r_t_2, len_1, len_2, labels
    
    def kcenter_sampling(self):

        # get queries
        num_init = self.mb_size*self.large_batch
        sa_t_1, sa_t_2, r_t_1, r_t_2, len_1, len_2 = self.get_queries(
            mb_size=num_init)

        if sa_t_1.shape[0] == 0:
            return 0

        # get final queries based on kmeans clustering
        temp_sa_t_1 = sa_t_1[:,:,:self.ds]
        temp_sa_t_2 = sa_t_2[:,:,:self.ds]
        temp_sa = np.concatenate([temp_sa_t_1.reshape(num_init, -1),
                                  temp_sa_t_2.reshape(num_init, -1)], axis=1)

        max_len = self.capacity if self.buffer_full else self.buffer_index

        if max_len == 0:
            selected_index = np.random.permutation(num_init)[:self.mb_size]
        else:
            tot_sa_1 = self.buffer_seg1[:max_len, :, :self.ds]
            tot_sa_2 = self.buffer_seg2[:max_len, :, :self.ds]
            tot_sa = np.concatenate([tot_sa_1.reshape(max_len, -1),
                                     tot_sa_2.reshape(max_len, -1)], axis=1)
            selected_index = KCenterGreedy(temp_sa, tot_sa, self.mb_size)

        r_t_1, sa_t_1 = r_t_1[selected_index], sa_t_1[selected_index]
        r_t_2, sa_t_2 = r_t_2[selected_index], sa_t_2[selected_index]
        len_1, len_2 = len_1[selected_index], len_2[selected_index]

        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, len_1, len_2, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2, len_1, len_2)

        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels, len_1, len_2)

        return len(labels)
    
    def kcenter_disagree_sampling(self):

        num_init = self.mb_size*self.large_batch
        num_init_half = int(num_init*0.5)

        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2, len_1, len_2 = self.get_queries(
            mb_size=num_init)

        if sa_t_1.shape[0] == 0:
            return 0

        # get final queries based on uncertainty
        _, disagree = self.get_rank_probability(sa_t_1, sa_t_2, len_1, len_2)
        top_k_index = (-disagree).argsort()[:num_init_half]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
        len_1, len_2 = len_1[top_k_index], len_2[top_k_index]

        # get final queries based on kmeans clustering
        temp_sa_t_1 = sa_t_1[:,:,:self.ds]
        temp_sa_t_2 = sa_t_2[:,:,:self.ds]

        temp_sa = np.concatenate([temp_sa_t_1.reshape(num_init_half, -1),
                                  temp_sa_t_2.reshape(num_init_half, -1)], axis=1)

        max_len = self.capacity if self.buffer_full else self.buffer_index

        if max_len == 0:
            selected_index = np.random.permutation(num_init_half)[:self.mb_size]
        else:
            tot_sa_1 = self.buffer_seg1[:max_len, :, :self.ds]
            tot_sa_2 = self.buffer_seg2[:max_len, :, :self.ds]
            tot_sa = np.concatenate([tot_sa_1.reshape(max_len, -1),
                                     tot_sa_2.reshape(max_len, -1)], axis=1)
            selected_index = KCenterGreedy(temp_sa, tot_sa, self.mb_size)

        r_t_1, sa_t_1 = r_t_1[selected_index], sa_t_1[selected_index]
        r_t_2, sa_t_2 = r_t_2[selected_index], sa_t_2[selected_index]
        len_1, len_2 = len_1[selected_index], len_2[selected_index]

        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, len_1, len_2, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2, len_1, len_2)

        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels, len_1, len_2)

        return len(labels)
    
    def kcenter_entropy_sampling(self):

        num_init = self.mb_size*self.large_batch
        num_init_half = int(num_init*0.5)

        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2, len_1, len_2 = self.get_queries(
            mb_size=num_init)

        if sa_t_1.shape[0] == 0:
            return 0

        # get final queries based on uncertainty
        entropy, _ = self.get_entropy(sa_t_1, sa_t_2, len_1, len_2)
        top_k_index = (-entropy).argsort()[:num_init_half]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
        len_1, len_2 = len_1[top_k_index], len_2[top_k_index]

        # get final queries based on kmeans clustering
        temp_sa_t_1 = sa_t_1[:,:,:self.ds]
        temp_sa_t_2 = sa_t_2[:,:,:self.ds]

        temp_sa = np.concatenate([temp_sa_t_1.reshape(num_init_half, -1),
                                  temp_sa_t_2.reshape(num_init_half, -1)], axis=1)

        max_len = self.capacity if self.buffer_full else self.buffer_index

        if max_len == 0:
            selected_index = np.random.permutation(num_init_half)[:self.mb_size]
        else:
            tot_sa_1 = self.buffer_seg1[:max_len, :, :self.ds]
            tot_sa_2 = self.buffer_seg2[:max_len, :, :self.ds]
            tot_sa = np.concatenate([tot_sa_1.reshape(max_len, -1),
                                     tot_sa_2.reshape(max_len, -1)], axis=1)
            selected_index = KCenterGreedy(temp_sa, tot_sa, self.mb_size)

        r_t_1, sa_t_1 = r_t_1[selected_index], sa_t_1[selected_index]
        r_t_2, sa_t_2 = r_t_2[selected_index], sa_t_2[selected_index]
        len_1, len_2 = len_1[selected_index], len_2[selected_index]

        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, len_1, len_2, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2, len_1, len_2)

        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels, len_1, len_2)

        return len(labels)
    
    def uniform_sampling(self):
        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2, len_1, len_2 = self.get_queries(
            mb_size=self.mb_size)

        if sa_t_1.shape[0] == 0:
            return 0

        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, len_1, len_2, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2, len_1, len_2)

        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels, len_1, len_2)

        return len(labels)
    
    def disagreement_sampling(self):

        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2, len_1, len_2 = self.get_queries(
            mb_size=self.mb_size*self.large_batch)

        if sa_t_1.shape[0] == 0:
            return 0

        # get final queries based on uncertainty
        _, disagree = self.get_rank_probability(sa_t_1, sa_t_2, len_1, len_2)
        top_k_index = (-disagree).argsort()[:self.mb_size]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
        len_1, len_2 = len_1[top_k_index], len_2[top_k_index]

        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, len_1, len_2, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2, len_1, len_2)
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels, len_1, len_2)

        return len(labels)
    
    def entropy_sampling(self):

        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2, len_1, len_2 = self.get_queries(
            mb_size=self.mb_size*self.large_batch)

        if sa_t_1.shape[0] == 0:
            return 0

        # get final queries based on uncertainty
        entropy, _ = self.get_entropy(sa_t_1, sa_t_2, len_1, len_2)

        top_k_index = (-entropy).argsort()[:self.mb_size]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
        len_1, len_2 = len_1[top_k_index], len_2[top_k_index]

        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, len_1, len_2, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2, len_1, len_2)

        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels, len_1, len_2)

        return len(labels)
    
    def train_reward(self):
        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.array([0 for _ in range(self.de)])
        
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = []
        for _ in range(self.de):
            total_batch_index.append(np.random.permutation(max_len))
        
        num_epochs = int(np.ceil(max_len/self.train_batch_size))
        list_debug_loss1, list_debug_loss2 = [], []
        total = 0
        
        for epoch in range(num_epochs):
            self.opt.zero_grad()
            loss = 0.0
            
            last_index = (epoch+1)*self.train_batch_size
            if last_index > max_len:
                last_index = max_len
                
            for member in range(self.de):
                
                # get random batch
                idxs = total_batch_index[member][epoch*self.train_batch_size:last_index]
                sa_t_1 = self.buffer_seg1[idxs]
                sa_t_2 = self.buffer_seg2[idxs]
                labels_np = self.buffer_label[idxs].flatten()
                len_1_t = torch.from_numpy(self.buffer_len1[idxs]).long().to(device)
                len_2_t = torch.from_numpy(self.buffer_len2[idxs]).long().to(device)

                if member == 0:
                    total += len(labels_np)

                # get logits
                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)
                r_hat1 = self._masked_sum_torch(r_hat1, len_1_t)
                r_hat2 = self._masked_sum_torch(r_hat2, len_2_t)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

                # compute loss — soft CE for fractional labels (e.g. 0.5)
                soft_mask = (labels_np != labels_np.astype(int).astype(float))
                if soft_mask.any():
                    # Build soft target distribution: [1-p, p] where p is label
                    soft_targets = torch.from_numpy(labels_np).float().to(device)
                    target_dist = torch.stack([1 - soft_targets, soft_targets], dim=-1)
                    log_probs = F.log_softmax(r_hat, dim=-1)
                    curr_loss = -(target_dist * log_probs).sum(dim=-1).mean()
                else:
                    labels_t = torch.from_numpy(labels_np.astype(int)).long().to(device)
                    curr_loss = self.CEloss(r_hat, labels_t)
                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())

                # compute acc (hard labels only for accuracy)
                _, predicted = torch.max(r_hat.data, 1)
                hard_labels = torch.from_numpy(np.round(labels_np).astype(int)).long().to(device)
                correct = (predicted == hard_labels).sum().item()
                ensemble_acc[member] += correct

            loss.backward()
            self.opt.step()

        ensemble_acc = ensemble_acc / total

        return ensemble_acc

    def train_soft_reward(self):
        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.array([0 for _ in range(self.de)])
        
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = []
        for _ in range(self.de):
            total_batch_index.append(np.random.permutation(max_len))
        
        num_epochs = int(np.ceil(max_len/self.train_batch_size))
        list_debug_loss1, list_debug_loss2 = [], []
        total = 0
        
        for epoch in range(num_epochs):
            self.opt.zero_grad()
            loss = 0.0
            
            last_index = (epoch+1)*self.train_batch_size
            if last_index > max_len:
                last_index = max_len
                
            for member in range(self.de):
                
                # get random batch
                idxs = total_batch_index[member][epoch*self.train_batch_size:last_index]
                sa_t_1 = self.buffer_seg1[idxs]
                sa_t_2 = self.buffer_seg2[idxs]
                labels = self.buffer_label[idxs]
                labels = torch.from_numpy(labels.flatten()).long().to(device)
                len_1_t = torch.from_numpy(self.buffer_len1[idxs]).long().to(device)
                len_2_t = torch.from_numpy(self.buffer_len2[idxs]).long().to(device)

                if member == 0:
                    total += labels.size(0)

                # get logits
                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)
                r_hat1 = self._masked_sum_torch(r_hat1, len_1_t)
                r_hat2 = self._masked_sum_torch(r_hat2, len_2_t)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

                # compute loss
                uniform_index = labels == -1
                labels[uniform_index] = 0
                target_onehot = torch.zeros_like(r_hat).scatter(1, labels.unsqueeze(1), self.label_target)
                target_onehot += self.label_margin
                if sum(uniform_index) > 0:
                    target_onehot[uniform_index] = 0.5
                curr_loss = self.softXEnt_loss(r_hat, target_onehot)
                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())

                # compute acc
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct

            loss.backward()
            self.opt.step()

        ensemble_acc = ensemble_acc / total

        return ensemble_acc

    # ------------------------------------------------------------------
    # Human-in-the-loop preference labeling
    # ------------------------------------------------------------------

    def human_sampling(self):
        """Present queries to a human in the terminal for preference labeling.

        Follows the same pattern as uniform_sampling / disagreement_sampling:
        get_queries -> get labels -> put_queries.
        """
        from human_label import get_human_labels

        sa_t_1, sa_t_2, r_t_1, r_t_2, len_1, len_2 = self.get_queries(
            mb_size=self.mb_size)

        if sa_t_1.shape[0] == 0:
            return 0

        labels, keep_indices = get_human_labels(
            sa_t_1, sa_t_2, len_1, len_2, self.ds)

        if len(labels) > 0:
            self.put_queries(
                sa_t_1[keep_indices], sa_t_2[keep_indices],
                labels,
                len_1[keep_indices], len_2[keep_indices])

        return len(labels)

    # ------------------------------------------------------------------
    # Save / load trajectory data for offline labeling
    # ------------------------------------------------------------------

    def save_trajectories(self, path):
        """Dump trajectory data so human_label.py can label offline."""
        import pickle as _pkl
        data = {
            'inputs': self.inputs,
            'targets': self.targets,
            'ds': self.ds,
            'da': self.da,
            'size_segment': self.size_segment,
        }
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'wb') as f:
            _pkl.dump(data, f)
        print(f"[reward_model] saved {len(self.inputs)} trajectories -> {path}")

    def load_human_labels(self, path):
        """Load labels produced by an offline human_label.py session."""
        import pickle as _pkl
        with open(path, 'rb') as f:
            data = _pkl.load(f)

        sa_t_1 = data['sa_t_1']
        sa_t_2 = data['sa_t_2']
        labels = data['labels']
        len_1  = data['len_1']
        len_2  = data['len_2']

        # Convert "equal" labels (-1) to 0.5 soft label
        equal_mask = (labels.flatten() == -1)
        n_equal = int(equal_mask.sum())
        labels[equal_mask] = 0.5

        self.put_queries(sa_t_1, sa_t_2, labels, len_1, len_2)
        n = len(labels)
        print(f"[reward_model] loaded {n} human labels from {path} "
              f"({n_equal} equal -> 0.5 soft labels)")
        return n

    # ------------------------------------------------------------------
    # Segment video rendering (on-demand at query time, no frame storage)
    # ------------------------------------------------------------------

    @staticmethod
    def _reverse_lunar_obs(obs):
        """Reverse LunarLander observation normalization → world coordinates.

        LunarLander source (gym/envs/box2d/lunar_lander.py lines 348-356):
            state[0] = (pos.x - W/2) / (W/2)
            state[1] = (pos.y - (helipad_y + LEG_DOWN/SCALE)) / (H/2)
            state[2] = vel.x * (W/2) / FPS
            state[3] = vel.y * (H/2) / FPS
            state[4] = angle
            state[5] = 20.0 * angularVelocity / FPS

        Constants: VIEWPORT_W=600, VIEWPORT_H=400, SCALE=30, FPS=50, LEG_DOWN=18
            W = VIEWPORT_W/SCALE = 20.0
            H = VIEWPORT_H/SCALE = 13.333...
            helipad_y = H/4 = 3.333...
        """
        W = 600.0 / 30.0           # 20.0
        H = 400.0 / 30.0           # 13.333...
        helipad_y = H / 4.0        # 3.333...
        LEG_DOWN_S = 18.0 / 30.0   # 0.6
        FPS = 50.0

        pos_x = obs[0] * (W / 2) + (W / 2)
        pos_y = obs[1] * (H / 2) + helipad_y + LEG_DOWN_S
        vel_x = obs[2] * FPS / (W / 2)
        vel_y = obs[3] * FPS / (H / 2)
        angle = obs[4]
        ang_vel = obs[5] * FPS / 20.0
        return pos_x, pos_y, vel_x, vel_y, angle, ang_vel

    @staticmethod
    def render_segment_video(segment, length, obs_dim, env):
        """Render a segment as a list of RGB frames by setting env state.

        Args:
            segment: (size_segment, obs_dim+act_dim) array of obs-action pairs
            length: actual number of valid steps in segment
            obs_dim: observation dimensionality
            env: a *reset* gym LunarLander env instance

        Returns:
            list of numpy uint8 frames (H, W, 3)
        """
        import math
        import numpy as np

        unwrapped = env.unwrapped
        lander = unwrapped.lander
        legs = unwrapped.legs

        SCALE = 30.0
        FPS = 50.0
        SIDE_ENGINE_AWAY = 12.0
        SIDE_ENGINE_HEIGHT = 14.0
        LEG_AWAY_S = 20.0 / SCALE   # 0.667
        LEG_DOWN_S = 18.0 / SCALE   # 0.6

        # Clear any leftover particles from previous renders
        unwrapped.particles = []

        # Use a separate RNG so particle dispersion doesn't disturb
        # the global numpy random state (which training relies on).
        _rng = np.random.RandomState(42)

        frames = []
        for t in range(int(length)):
            obs = segment[t, :obs_dim]
            act = segment[t, obs_dim:]
            px, py, vx, vy, angle, ang_vel = RewardModel._reverse_lunar_obs(obs)

            # Set lander body state from observation
            lander.position.x = px
            lander.position.y = py
            lander.linearVelocity.x = vx
            lander.linearVelocity.y = vy
            lander.angle = float(angle)
            lander.angularVelocity = float(ang_vel)

            # Approximately position legs to follow the lander
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            for idx, sign in enumerate([-1, +1]):
                ax = sign * LEG_AWAY_S
                ay = LEG_DOWN_S
                legs[idx].position.x = px - (ax * cos_a - ay * sin_a)
                legs[idx].position.y = py - (ax * sin_a + ay * cos_a)
                legs[idx].angle = float(angle)

            # --- Create thruster particles from recorded actions ---
            tip = (math.sin(angle), math.cos(angle))
            side = (-tip[1], tip[0])
            disp = [_rng.uniform(-1.0, 1.0) / SCALE for _ in range(2)]

            # Main engine (action[0] > 0)
            if len(act) > 0 and act[0] > 0.0:
                m_power = (float(np.clip(act[0], 0.0, 1.0)) + 1.0) * 0.5
                ox = tip[0] * (4/SCALE + 2*disp[0]) + side[0] * disp[1]
                oy = -tip[1] * (4/SCALE + 2*disp[0]) - side[1] * disp[1]
                imp_x, imp_y = px + ox, py + oy
                p = unwrapped._create_particle(3.5, imp_x, imp_y, m_power)
                p.ApplyLinearImpulse(
                    (ox * 13.0 * m_power, oy * 13.0 * m_power),
                    (imp_x, imp_y), True)

            # Side engines (|action[1]| > 0.5)
            if len(act) > 1 and abs(act[1]) > 0.5:
                direction = float(np.sign(act[1]))
                s_power = float(np.clip(abs(act[1]), 0.5, 1.0))
                ox = (tip[0] * disp[0]
                      + side[0] * (3*disp[1] + direction * SIDE_ENGINE_AWAY/SCALE))
                oy = (-tip[1] * disp[0]
                      - side[1] * (3*disp[1] + direction * SIDE_ENGINE_AWAY/SCALE))
                imp_x = px + ox - tip[0] * 17/SCALE
                imp_y = py + oy + tip[1] * SIDE_ENGINE_HEIGHT/SCALE
                p = unwrapped._create_particle(0.7, imp_x, imp_y, s_power)
                p.ApplyLinearImpulse(
                    (ox * 0.6 * s_power, oy * 0.6 * s_power),
                    (imp_x, imp_y), True)

            # Step physics so particles drift visually
            unwrapped.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

            # Override lander/leg positions back (physics step moved them)
            lander.position.x = px
            lander.position.y = py
            lander.linearVelocity.x = vx
            lander.linearVelocity.y = vy
            lander.angle = float(angle)
            lander.angularVelocity = float(ang_vel)
            for idx, sign in enumerate([-1, +1]):
                ax = sign * LEG_AWAY_S
                ay = LEG_DOWN_S
                legs[idx].position.x = px - (ax * cos_a - ay * sin_a)
                legs[idx].position.y = py - (ax * sin_a + ay * cos_a)
                legs[idx].angle = float(angle)

            frame = env.render(mode='rgb_array')
            frames.append(frame)

        return frames

    def save_query_batch(self, sa_t_1, sa_t_2, r_t_1, r_t_2,
                         len_1, len_2, oracle_labels,
                         batch_idx, save_dir, env):
        """Render and save segment pair videos + metadata for offline labeling.

        Args:
            sa_t_1, sa_t_2: (batch, seg_len, obs+act) segment arrays
            r_t_1, r_t_2: (batch, seg_len, 1) reward arrays
            len_1, len_2: (batch,) actual segment lengths
            oracle_labels: (batch, 1) oracle labels or None
            batch_idx: int, batch counter for directory naming
            save_dir: root directory for query videos
            env: a reset gym LunarLander env instance for rendering
        """
        import pickle as _pkl
        import imageio

        batch_dir = os.path.join(save_dir, f'batch_{batch_idx:03d}')
        os.makedirs(batch_dir, exist_ok=True)

        n_pairs = sa_t_1.shape[0]
        for i in range(n_pairs):
            pair_dir = os.path.join(batch_dir, f'pair_{i:03d}')
            os.makedirs(pair_dir, exist_ok=True)
            try:
                frames_a = self.render_segment_video(
                    sa_t_1[i], len_1[i], self.ds, env)
                frames_b = self.render_segment_video(
                    sa_t_2[i], len_2[i], self.ds, env)
                if frames_a:
                    imageio.mimwrite(
                        os.path.join(pair_dir, 'seg_A.mp4'),
                        frames_a, fps=25)
                if frames_b:
                    imageio.mimwrite(
                        os.path.join(pair_dir, 'seg_B.mp4'),
                        frames_b, fps=25)
            except (IOError, OSError, Exception) as e:
                print(f"[save_query_batch] WARNING: video I/O failed for "
                      f"batch {batch_idx} pair {i}: {e}")

        # Save metadata pickle (exact segment data for deterministic offline labeling)
        meta = {
            'sa_t_1': sa_t_1,
            'sa_t_2': sa_t_2,
            'r_t_1': r_t_1,
            'r_t_2': r_t_2,
            'len_1': len_1,
            'len_2': len_2,
            'oracle_labels': oracle_labels,
            'obs_dim': self.ds,
            'act_dim': self.da,
            'size_segment': self.size_segment,
            'n_pairs': n_pairs,
            'batch_idx': batch_idx,
        }
        meta_path = os.path.join(batch_dir, 'metadata.pkl')
        with open(meta_path, 'wb') as f:
            _pkl.dump(meta, f)

        print(f"[save_query_batch] saved {n_pairs} pairs -> {batch_dir}")

    # ------------------------------------------------------------------
    # Online web-based human labeling (feed_type=7)
    # ------------------------------------------------------------------

    def web_sampling_replay(self, replay_dir):
        """Load a previously saved web batch from disk and apply its labels
        directly. Bypasses get_queries() — uses the EXACT segments and labels
        stored at original-run time, so the preference buffer ends up
        bit-identical to the original run.

        Returns the number of labeled pairs (or -1 if no batch is available
        at this batch_idx, signalling 'fall through to live web_sampling').
        """
        import pickle as _pkl

        batch_idx = self._online_batch_idx
        batch_dir = os.path.join(replay_dir, f'batch_{batch_idx:03d}')
        meta_path = os.path.join(batch_dir, 'metadata.pkl')
        resp_path = os.path.join(batch_dir, 'response.pkl')

        if not (os.path.exists(meta_path) and os.path.exists(resp_path)):
            print(f"[web_replay] no saved batch_{batch_idx:03d} in {replay_dir} "
                  f"-> falling through to live web_sampling")
            return -1

        with open(meta_path, 'rb') as f:
            meta = _pkl.load(f)
        with open(resp_path, 'rb') as f:
            resp = _pkl.load(f)

        sa_t_1 = meta['sa_t_1']
        sa_t_2 = meta['sa_t_2']
        len_1 = meta['len_1']
        len_2 = meta['len_2']
        labels = resp['labels']
        keep_indices = resp['keep_indices']
        human_time = float(resp.get('time_sec', 0.0))
        self._last_web_human_time = human_time
        self._last_web_wait_sec = float(resp.get('wait_for_start_sec', 0.0))

        n_labeled = len(labels)
        print(f"[web_replay] loaded batch_{batch_idx:03d}: "
              f"{n_labeled} labels, human_time={human_time:.1f}s "
              f"wait_for_start={self._last_web_wait_sec:.1f}s")

        if n_labeled > 0:
            self.put_queries(
                sa_t_1[keep_indices], sa_t_2[keep_indices],
                labels, len_1[keep_indices], len_2[keep_indices])

        self._online_batch_idx += 1
        return n_labeled

    def web_sampling(self, online_query_dir, env, poll_interval=2.0, timeout=3600):
        """Generate queries, render videos, wait for web-based human labels.

        Writes videos + request.json to online_query_dir. Polls for
        response.pkl written by label_web.py. Returns number of labeled pairs.
        """
        import json
        import pickle as _pkl

        # 1. Get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2, len_1, len_2 = self.get_queries(
            mb_size=self.mb_size)

        if sa_t_1.shape[0] == 0:
            return 0

        batch_idx = self._online_batch_idx
        batch_dir = os.path.join(online_query_dir, f'batch_{batch_idx:03d}')
        os.makedirs(batch_dir, exist_ok=True)

        # 2. Render and save segment videos
        n_pairs = sa_t_1.shape[0]
        print(f"[web_sampling] rendering {n_pairs} pairs for batch {batch_idx}...")

        for i in range(n_pairs):
            pair_dir = os.path.join(batch_dir, f'pair_{i:03d}')
            os.makedirs(pair_dir, exist_ok=True)
            try:
                import imageio
                frames_a = self.render_segment_video(
                    sa_t_1[i], len_1[i], self.ds, env)
                frames_b = self.render_segment_video(
                    sa_t_2[i], len_2[i], self.ds, env)
                if frames_a:
                    imageio.mimwrite(
                        os.path.join(pair_dir, 'seg_A.mp4'), frames_a, fps=25)
                if frames_b:
                    imageio.mimwrite(
                        os.path.join(pair_dir, 'seg_B.mp4'), frames_b, fps=25)
            except Exception as e:
                print(f"[web_sampling] video render failed pair {i}: {e}")

        # 3. Save metadata
        meta = {
            'sa_t_1': sa_t_1, 'sa_t_2': sa_t_2,
            'r_t_1': r_t_1, 'r_t_2': r_t_2,
            'len_1': len_1, 'len_2': len_2,
            'obs_dim': self.ds, 'act_dim': self.da,
            'size_segment': self.size_segment,
            'n_pairs': n_pairs, 'batch_idx': batch_idx,
        }
        with open(os.path.join(batch_dir, 'metadata.pkl'), 'wb') as f:
            _pkl.dump(meta, f)

        # 4. Write request.json LAST (signals "ready for labeling")
        request = {
            'batch_idx': batch_idx,
            'n_pairs': n_pairs,
            'timestamp': time.time(),
            'status': 'pending',
        }
        req_path = os.path.join(batch_dir, 'request.json')
        tmp_path = req_path + '.tmp'
        with open(tmp_path, 'w') as f:
            json.dump(request, f)
        os.rename(tmp_path, req_path)

        print(f"[web_sampling] batch {batch_idx} ready ({n_pairs} pairs). "
              f"Waiting for labels from web UI...")

        # 5. Poll for response.pkl
        resp_path = os.path.join(batch_dir, 'response.pkl')
        t0 = time.time()
        while not os.path.exists(resp_path):
            elapsed = time.time() - t0
            if elapsed > timeout:
                raise TimeoutError(
                    f"No labels received for batch {batch_idx} after {timeout}s")
            mins = int(elapsed) // 60
            secs = int(elapsed) % 60
            print(f"\r[web_sampling] waiting for labels... {mins}m{secs:02d}s",
                  end='', flush=True)
            time.sleep(poll_interval)
        print()  # newline after \r

        # 6. Load response
        time.sleep(0.5)  # brief delay to ensure file write is complete
        with open(resp_path, 'rb') as f:
            response = _pkl.load(f)

        labels = response['labels']
        keep_indices = response['keep_indices']
        human_time = response.get('time_sec', 0.0)
        self._last_web_human_time = float(human_time)
        self._last_web_wait_sec = float(response.get('wait_for_start_sec', 0.0))

        n_labeled = len(labels)
        print(f"[web_sampling] received {n_labeled} labels for batch {batch_idx} "
              f"(human_time={human_time:.1f}s wait_for_start={self._last_web_wait_sec:.1f}s)")

        # 7. Store labeled queries
        if n_labeled > 0:
            self.put_queries(
                sa_t_1[keep_indices], sa_t_2[keep_indices],
                labels, len_1[keep_indices], len_2[keep_indices])

        self._online_batch_idx += 1
        return n_labeled