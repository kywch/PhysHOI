# from pdb import set_trace as T
import copy
import numpy as np

# isaacgym must be imported before torch
import isaacgym
import torch

from rl_games.common import env_configurations

from physhoi.utils.config import set_np_formatting, get_args, load_cfg
from physhoi.learning import physhoi_agent as org_physhoi_agent
from physhoi.learning import physhoi_players as org_physhoi_players
import physhoi.learning.rlgpu

from physhoi.norlg_learning import physhoi_agent
from physhoi.norlg_learning import physhoi_players
from physhoi.norlg_learning.utils import RLGPUAlgoObserver, DefaultRewardsShaper

from physhoi.norlg_learning.env import create_rlgpu_env, RLGPUEnvWrapper
from physhoi.norlg_learning.network import PhysHOINetworkBuilder, PhysHOIModelBuilder

RUN_EVAL = False
RUN_RLG = False


# Replace rlgames' torch_runner and factories
class Runner:
    def __init__(self, env_creator, algo_observer=None):
        self.env_creator = env_creator
        self.algo_observer = algo_observer
        torch.backends.cudnn.benchmark = True

    def reset(self):
        pass

    def load(self, yaml_conf):
        self.default_config = yaml_conf['params']
        self.load_config(copy.deepcopy(self.default_config))

        # if 'experiment_config' in yaml_conf:
        #     self.exp_config = yaml_conf['experiment_config']

    def load_config(self, params):  # params = cfg_train
        self.seed = params.get('seed', None)

        self.algo_params = params['algo']
        self.algo_name = self.algo_params['name']
        self.load_check_point = params['load_checkpoint']
        self.exp_config = None

        if self.seed:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            np.random.seed(self.seed)

        if self.load_check_point:
            print('Found checkpoint')
            print(params['load_path'])
            self.load_path = params['load_path']

        # self.model is actually model builder, not the model itself
        self.model = self.make_model_builder(params)
        self.config = copy.deepcopy(params['config'])
        
        self.config['reward_shaper'] = DefaultRewardsShaper(**self.config['reward_shaper'])
        self.config['network'] = self.model

    def make_model_builder(self, params):
        # network_builder = physhoi_network_builder.PhysHOIBuilder()
        network_builder = PhysHOINetworkBuilder()
        network_builder.load(params["network"])
        model_builder = PhysHOIModelBuilder(network_builder)
        return model_builder

    def run(self, args):
        if 'checkpoint' in args and args['checkpoint'] is not None:
            if len(args['checkpoint']) > 0:
                self.load_path = args['checkpoint']

        if args['train']:
            self.run_train()

        elif args['play']:
            print('Started to play')
            player = self.create_player()
            player.restore(self.load_path)
            player.run()

        else:
            raise ValueError(f'Unknown command: {args}')

    # "player" seems to be inference-only mode
    def create_player(self):
        if RUN_RLG:
            return org_physhoi_players.PhysHOIPlayerContinuous(self.config)
        else:
            return physhoi_players.PhysHOIPlayerContinuous(self.config, self.env_creator)

    def run_train(self):
        print('Started to train')
        self.reset()
        self.load_config(self.default_config)

        if self.algo_observer is None:
            self.algo_observer = RLGPUAlgoObserver()
        self.config['algo_observer'] = self.algo_observer

        if RUN_RLG:
            # for the org_physhoi_agent
            self.config['features'] = {"observer": self.algo_observer}
            agent = org_physhoi_agent.PhysHOIAgent(base_name='run', config=self.config)

        else:
            vec_env = self.env_creator()
            vec_env = RLGPUEnvWrapper(vec_env)
            agent = physhoi_agent.PhysHOIAgent(self.config, vec_env)

        if self.load_check_point and (self.load_path is not None):
            agent.restore(self.load_path)

        # CHECK ME: is resume_from necessary?
        # if agent.resume_from != 'None':
        #     agent.restore(self.resume_from)

        agent.train()


if __name__ == '__main__':
    set_np_formatting()
    args = get_args()

    if RUN_EVAL:
        # CLI arguments for eval
        args.test = True
        args.task = "PhysHOI_BallPlay"
        args.num_envs = 4
        args.cfg_env = "physhoi/data/cfg/physhoi.yaml"
        args.cfg_train = "physhoi/data/cfg/train/rlg/physhoi.yaml"
        args.motion_file = "physhoi/data/motions/BallPlay/backdribble.pt"
        args.checkpoint = "physhoi/data/models/backdribble/nn/PhysHOI.pth"

    else:
        # CLI arguments for train
        args.task = "PhysHOI_BallPlay"
        args.cfg_env = "physhoi/data/cfg/physhoi.yaml"
        args.cfg_train = "physhoi/data/cfg/train/rlg/physhoi.yaml"
        args.motion_file = "physhoi/data/motions/BallPlay/backdribble.pt"
        args.headless = True

    # Set the correct mode
    if args.test:
        args.play = args.test
        args.train = False
    elif args.play:
        args.train = False
    else:
        args.train = True

    vargs = vars(args)

    # config processing
    cfg, cfg_train, logdir = load_cfg(args)
    cfg['env']['motion_file'] = args.motion_file

    env_creator = lambda **kwargs: create_rlgpu_env(args, cfg, cfg_train, **kwargs)

    # Create and register the env creator with args, cfg, cfg_train
    env_configurations.register('rlgpu', {
        'env_creator': env_creator,
        'vecenv_type': 'RLGPU'}
    )

    runner = Runner(env_creator)
    runner.load(cfg_train)
    runner.reset()

    runner.run(vargs)
