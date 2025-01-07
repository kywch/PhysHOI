from gym import spaces
import numpy as np
import torch


# This wrapper combines VecTask, VecTaskPython, VecTaskPythonWrapper, RLGPUEnvWrapper
# Also does the action clipping
# CHECK ME: How about running mean norm on the observations?
class VecTaskWrapper:
    def __init__(self, task, rl_device, clip_observations=5.0, clip_actions=1.0):
        self.task = task

        self.num_environments = task.num_envs
        self.num_agents = 1  # used for multi-agent environments
        self.num_observations = task.num_obs
        # print("task.num_obs",task.num_obs)
        self.num_states = task.num_states
        self.num_actions = task.num_actions

        self.obs_space = spaces.Box(np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf)
        self.state_space = spaces.Box(
            np.ones(self.num_states) * -np.Inf, np.ones(self.num_states) * np.Inf
        )
        self.act_space = spaces.Box(
            np.ones(self.num_actions) * -1.0, np.ones(self.num_actions) * 1.0
        )

        self.clip_obs = clip_observations
        self.clip_actions = clip_actions
        self.rl_device = rl_device

        print("RL device: ", rl_device)

        # RLGPU env wrapper
        self.use_global_obs = self.task.num_states > 0

        self.full_state = {}
        self.full_state["obs"] = self.reset()
        if self.use_global_obs:
            self.full_state["states"] = self.task.get_state()

        # AMP-related
        self._amp_obs_space = spaces.Box(
            np.ones(task.get_num_amp_obs()) * -np.Inf, np.ones(task.get_num_amp_obs()) * np.Inf
        )

    @property
    def observation_space(self):
        return self.obs_space

    @property
    def action_space(self):
        return self.act_space

    @property
    def amp_observation_space(self):
        return self._amp_obs_space

    @property
    def num_envs(self):
        return self.num_environments

    @property
    def num_acts(self):
        return self.num_actions

    @property
    def num_obs(self):
        return self.num_observations

    @property
    def gym(self):
        return self.task.gym

    @property
    def viewer(self):
        return self.task.viewer

    def fetch_amp_obs_demo(self, num_samples):
        return self.task.fetch_amp_obs_demo(num_samples)

    def get_state(self):
        return torch.clamp(self.task.states_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

    def _process_obs(self, obs):
        return torch.clamp(obs, -self.clip_obs, self.clip_obs).to(self.rl_device)

    def reset(self, env_ids=None):
        self.task.reset(env_ids)
        self.full_state["obs"] = self._process_obs(self.task.obs_buf)

        if self.use_global_obs:
            self.full_state["states"] = self.task.get_state()
            return self.full_state
        else:
            return self.full_state["obs"]

    def step(self, actions):
        # Action clipping
        actions_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)

        self.task.step(actions_tensor)

        next_obs = self._process_obs(self.task.obs_buf)
        rewards = self.task.rew_buf.to(self.rl_device)
        dones = self.task.reset_buf.to(self.rl_device)
        infos = self.task.extras

        self.full_state["obs"] = next_obs
        if self.use_global_obs:
            self.full_state["states"] = self.task.get_state()
            return self.full_state, rewards, dones, infos
        else:
            return self.full_state["obs"], rewards, dones, infos
