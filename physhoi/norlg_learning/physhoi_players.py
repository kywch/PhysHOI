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

import os
import time

import gym
import torch

from .env import get_env_info
from .utils import RunningMeanStd, shape_whc_to_cwh, rescale_actions


class PhysHOIAgent:
    def __init__(self, config, env):
        # from rl_games.common.player import BasePlayer
        # BasePlayer.__init__(self, config)
        self.config = config
        # self.env_name = self.config['env_name']
        # self.env_config = self.config.get('env_config', {})
        self.env_info = self.config.get("env_info")
        self.clip_actions = config.get("clip_actions", True)

        self.env = env
        if self.env_info is None:
            # self.env = env_creator(**self.env_config)  # self.create_env()
            self.env_info = get_env_info(self.env)

        self.value_size = self.env_info.get("value_size", 1)
        self.action_space = self.env_info["action_space"]
        self.num_agents = self.env_info["agents"]

        self.observation_space = self.env_info["observation_space"]
        if isinstance(self.observation_space, gym.spaces.Dict):
            self.obs_shape = {}
            for k, v in self.observation_space.spaces.items():
                self.obs_shape[k] = v.shape
        else:
            self.obs_shape = self.observation_space.shape

        self.states = None
        self.player_config = self.config.get("player", {})
        self.batch_size = 1
        # self.has_central_value = self.config.get('central_value_config') is not None
        self.render_env = self.player_config.get("render", False)
        self.games_num = self.player_config.get("games_num", 15)
        self.is_determenistic = self.player_config.get("determenistic", True)
        self.print_stats = self.player_config.get("print_stats", True)
        self.render_sleep = self.player_config.get("render_sleep", 0.002)
        self.max_steps = 108000 // 4

        # TODO: check device in config. For now, there is no device nor device_name
        self.use_cuda = True
        self.device_name = self.config.get("device_name", "cuda")
        self.device = torch.device(self.device_name)

        # from physhoi.learning import common_player
        # common_player.CommonPlayer.__init__(self, config)
        self.network = config["network"]

        self._setup_action_space()
        self.mask = [False]

        self.normalize_input = self.config.get("normalize_input", False)
        # CHECK ME: normalize_value is not used in PhysHOI checkpoints
        self.normalize_value = False  # self.config.get('normalize_value', False)
        self._build_model()

        # Adversarial Motion Prior (AMP)-related
        self._normalize_amp_input = config.get("normalize_amp_input", True)
        self._amp_input_mean_std = None
        if self._normalize_amp_input:
            assert hasattr(self, "env"), "env is not set"
            config["amp_input_shape"] = self.env.amp_observation_space.shape
            self._amp_input_mean_std = RunningMeanStd(config["amp_input_shape"]).to(self.device)
            self._amp_input_mean_std.eval()

    def _setup_action_space(self):
        self.actions_num = self.action_space.shape[0]
        self.actions_low = torch.from_numpy(self.action_space.low.copy()).float().to(self.device)
        self.actions_high = torch.from_numpy(self.action_space.high.copy()).float().to(self.device)

    def _build_model(self):
        # net_config = self._build_net_config()
        obs_shape = shape_whc_to_cwh(self.obs_shape)
        config = {
            "actions_num": self.actions_num,
            "input_shape": obs_shape,
            # 'num_seqs' : self.num_agents  # used for rnn, so not needed
        }

        # self._build_net(net_config)
        self.model = self.network.build(config)
        self.model.to(self.device)
        self.model.eval()
        self.is_rnn = self.model.is_rnn()
        assert not self.is_rnn, "PhysHOI is not supported with RNN"

        if self.normalize_input:
            self.running_mean_std = RunningMeanStd(obs_shape).to(self.device)
            self.running_mean_std.eval()

        # if self.normalize_value:
        #     self.value_mean_std = RunningMeanStd((self.value_size,)).to(self.device)
        #     self.value_mean_std.eval()

    def set_eval(self):
        self.model.eval()
        if self.normalize_input:
            self.running_mean_std.eval()
        if self._normalize_amp_input:
            self._amp_input_mean_std.eval()

    def set_train(self):
        self.model.train()
        if self.normalize_input:
            self.running_mean_std.train()
        if self._normalize_amp_input:
            self._amp_input_mean_std.train()

    def get_model_weights(self):
        state_dict = {}
        state_dict["model"] = self.model.state_dict()
        if self.normalize_input:
            state_dict["running_mean_std"] = self.running_mean_std.state_dict()
        if self._normalize_amp_input:
            state_dict["amp_input_mean_std"] = self._amp_input_mean_std.state_dict()
        return state_dict

    def set_model_weights(self, state_dict):
        self.model.load_state_dict(state_dict["model"])
        if self.normalize_input:
            self.running_mean_std.load_state_dict(state_dict["running_mean_std"])
        if self._normalize_amp_input:
            self._amp_input_mean_std.load_state_dict(state_dict["amp_input_mean_std"])

    def restore(self, file_path):
        if os.path.exists(file_path):
            print("=> loading checkpoint '{}'".format(file_path))
            state_dict = torch.load(file_path)
            self.set_model_weights(state_dict)

    def env_reset(self, env_ids=None):
        obs_torch = self.env.reset(env_ids)
        # obs is already in torch
        return obs_torch

    def _preproc_obs(self, obs_batch):
        # if type(obs_batch) is dict:
        #     for k, v in obs_batch.items():
        #         obs_batch[k] = self._preproc_obs(v)
        # else:
        #     if obs_batch.dtype == torch.uint8:
        #         obs_batch = obs_batch.float() / 255.0
        if self.normalize_input:
            obs_batch = self.running_mean_std(obs_batch)
        return obs_batch


class PhysHOIPlayerContinuous(PhysHOIAgent):
    def __init__(self, config, env_creator):
        env_config = config.get("env_config", {})
        env = env_creator(**env_config)

        super().__init__(config, env)

    def get_batch_size(self, obses):
        obs_shape = self.obs_shape
        assert len(obses.size()) > len(obs_shape), "obses must be batched"
        self.batch_size = obses.size()[0]
        return self.batch_size

    def get_action(self, obs_torch, is_determenistic=False):
        obs_torch = self._preproc_obs(obs_torch)
        input_dict = {
            "is_train": False,
            "prev_actions": None,
            "obs": obs_torch,
            "rnn_states": self.states,
        }

        with torch.no_grad():
            res_dict = self.model(input_dict)
        mu = res_dict["mus"]
        action = res_dict["actions"]

        if is_determenistic:
            current_action = mu
        else:
            current_action = action

        if not self.clip_actions:
            return current_action

        return rescale_actions(
            self.actions_low, self.actions_high, torch.clamp(current_action, -1.0, 1.0)
        )

    def run(self):
        n_games = self.games_num
        render = self.render_env
        is_determenistic = self.is_determenistic
        sum_rewards = 0
        sum_steps = 0
        games_played = 0

        for _ in range(n_games):
            if games_played >= n_games:
                break

            obs_torch = self.env_reset()
            batch_size = self.get_batch_size(obs_torch)

            cr = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            steps = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

            done_indices = []

            if self.env.task.play_dataset:
                # play dataset
                while True:
                    for t in range(self.env.task.max_episode_length):
                        self.env.task.play_dataset_step(t)

            else:
                # inference
                for _ in range(self.max_steps):
                    obs_torch = self.env_reset(done_indices)

                    action = self.get_action(obs_torch, is_determenistic)

                    """Stepping the environment"""
                    obs_torch, r, done, info = self.env.step(action)

                    cr += r
                    steps += 1

                    self._post_step(info)

                    if render:
                        self.env.render(mode="human")
                        time.sleep(self.render_sleep)

                    all_done_indices = done.nonzero(as_tuple=False)
                    done_indices = all_done_indices[:: self.num_agents]
                    done_count = len(done_indices)
                    games_played += done_count

                    if done_count > 0:
                        cur_rewards = cr[done_indices].sum().item()
                        cur_steps = steps[done_indices].sum().item()

                        cr = cr * (1.0 - done.float())
                        steps = steps * (1.0 - done.float())
                        sum_rewards += cur_rewards
                        sum_steps += cur_steps

                        if self.print_stats:
                            print(
                                "games_played:",
                                games_played,
                                "reward:",
                                cur_rewards / done_count,
                                "steps:",
                                cur_steps / done_count,
                            )

                        if batch_size // self.num_agents == 1 or games_played >= n_games:
                            break

                    done_indices = done_indices[:, 0]

        print(n_games, "games played. Done.")
        return

    def _post_step(self, info):
        if self.env.task.viewer:
            self._amp_debug(info)
        return

    def _amp_debug(self, info):
        return
