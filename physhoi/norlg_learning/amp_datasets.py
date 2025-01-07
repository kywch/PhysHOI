# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
from torch.utils.data import Dataset


class PPODataset(Dataset):
    def __init__(self, batch_size, minibatch_size, device):
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.device = device
        self.length = self.batch_size // self.minibatch_size

        self.special_names = ["rnn_states"]

    def update_values_dict(self, values_dict):
        self.values_dict = values_dict

    def update_mu_sigma(self, mu, sigma):
        start = self.last_range[0]
        end = self.last_range[1]
        self.values_dict["mu"][start:end] = mu
        self.values_dict["sigma"][start:end] = sigma

    def __len__(self):
        return self.length

    def _get_item(self, idx):
        start = idx * self.minibatch_size
        end = (idx + 1) * self.minibatch_size
        self.last_range = (start, end)
        input_dict = {}
        for k, v in self.values_dict.items():
            if k not in self.special_names and v is not None:
                if type(v) is dict:
                    v_dict = {kd: vd[start:end] for kd, vd in v.items()}
                    input_dict[k] = v_dict
                else:
                    input_dict[k] = v[start:end]

        return input_dict

    def __getitem__(self, idx):
        return self._get_item(idx)


class AMPDataset(PPODataset):
    def __init__(self, batch_size, minibatch_size, device):
        super().__init__(batch_size, minibatch_size, device)
        self._idx_buf = torch.randperm(batch_size)

    def update_mu_sigma(self, mu, sigma):
        raise NotImplementedError()

    def _get_item(self, idx):
        start = idx * self.minibatch_size
        end = (idx + 1) * self.minibatch_size
        sample_idx = self._idx_buf[start:end]

        input_dict = {}
        for k, v in self.values_dict.items():
            if k not in self.special_names and v is not None:
                input_dict[k] = v[sample_idx]

        if end >= self.batch_size:
            self._shuffle_idx_buf()

        return input_dict

    def _shuffle_idx_buf(self):
        self._idx_buf[:] = torch.randperm(self.batch_size)
        return
