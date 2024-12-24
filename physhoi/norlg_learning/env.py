import numpy as np

from physhoi.utils.config import parse_sim_params
from physhoi.env.tasks.physhoi import PhysHOI_BallPlay
from physhoi.env.tasks.vec_task_wrappers import VecTaskPythonWrapper

from rl_games.common import env_configurations, vecenv

vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))


def create_rlgpu_env(args, cfg, cfg_train, **kwargs):

    sim_params = parse_sim_params(args, cfg, cfg_train)

    # create native task and pass custom config
    device_id = args.device_id
    rl_device = args.rl_device

    cfg["seed"] = cfg_train.get("seed", 0)
    cfg_task = cfg["env"]
    cfg_task["seed"] = cfg["seed"]

    try:
        task = PhysHOI_BallPlay(
            cfg=cfg,
            sim_params=sim_params,
            physics_engine=args.physics_engine,
            device_type=args.device,
            device_id=device_id,
            headless=args.headless
        )

    except NameError as e:
        print(e)

    env = VecTaskPythonWrapper(task, rl_device, cfg_train.get("clip_observations", np.inf), cfg_train.get("clip_actions", 1.0))

    print('num_envs: {:d}'.format(env.num_envs))
    print('num_actions: {:d}'.format(env.num_actions))
    print('num_obs: {:d}'.format(env.num_obs))
    print('num_states: {:d}'.format(env.num_states))
    
    return env


# If there are missing methods, see rl_games/common/ivecenv.py: class IVecEnv
class RLGPUEnv:
    def __init__(self, config_name, num_actors, **kwargs):
        self.env = env_configurations.configurations[config_name]['env_creator'](**kwargs)
        # pdb.set_trace()
        self.use_global_obs = (self.env.num_states > 0)

        self.full_state = {}
        self.full_state["obs"] = self.reset()
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
        return

    def step(self, action):
        next_obs, reward, is_done, info = self.env.step(action)

        # todo: improve, return only dictinary
        self.full_state["obs"] = next_obs
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
            return self.full_state, reward, is_done, info
        else:
            return self.full_state["obs"], reward, is_done, info

    def reset(self, env_ids=None):
        self.full_state["obs"] = self.env.reset(env_ids)
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
            return self.full_state
        else:
            return self.full_state["obs"]

    def get_number_of_agents(self):
        return self.env.get_number_of_agents()

    def get_env_info(self):
        info = {}
        info['action_space'] = self.env.action_space
        info['observation_space'] = self.env.observation_space
        info['amp_observation_space'] = self.env.amp_observation_space

        if self.use_global_obs:
            info['state_space'] = self.env.state_space
            print(info['action_space'], info['observation_space'], info['state_space'])
        else:
            print(info['action_space'], info['observation_space'])

        return info


def get_env_info(env):
    result_shapes = {}
    result_shapes['observation_space'] = env.observation_space
    result_shapes['action_space'] = env.action_space
    result_shapes['agents'] = 1
    result_shapes['value_size'] = 1
    if hasattr(env, "get_number_of_agents"):
        result_shapes['agents'] = env.get_number_of_agents()
    '''
    if isinstance(result_shapes['observation_space'], gym.spaces.dict.Dict):
        result_shapes['observation_space'] = observation_space['observations']
    if isinstance(result_shapes['observation_space'], dict):
        result_shapes['observation_space'] = observation_space['observations']
        result_shapes['state_space'] = observation_space['states']
    '''
    if hasattr(env, "value_size"):    
        result_shapes['value_size'] = env.value_size
    print(result_shapes)
    return result_shapes
