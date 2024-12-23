# from pdb import set_trace as T
import copy
import numpy as np

# isaacgym must be imported before torch
import isaacgym
import torch

from rl_games.common import env_configurations
from rl_games.algos_torch import torch_ext
from rl_games.common.algo_observer import AlgoObserver
from rl_games.algos_torch.models import ModelA2CContinuousLogStd

from physhoi.utils.config import set_np_formatting, get_args, load_cfg
from physhoi.learning import physhoi_agent
from physhoi.learning import physhoi_players
from physhoi.learning import physhoi_network_builder

from env import create_rlgpu_env


RUN_EVAL = True


class RLGPUAlgoObserver(AlgoObserver):
    def __init__(self, use_successes=True):
        self.use_successes = use_successes
        return

    def after_init(self, algo):
        self.algo = algo
        self.consecutive_successes = torch_ext.AverageMeter(1, self.algo.games_to_track).to(self.algo.ppo_device)
        self.writer = self.algo.writer
        return

    def process_infos(self, infos, done_indices):
        if isinstance(infos, dict):
            if (self.use_successes == False) and 'consecutive_successes' in infos:
                cons_successes = infos['consecutive_successes'].clone()
                self.consecutive_successes.update(cons_successes.to(self.algo.ppo_device))
            if self.use_successes and 'successes' in infos:
                successes = infos['successes'].clone()
                self.consecutive_successes.update(successes[done_indices].to(self.algo.ppo_device))
        return

    def after_clear_stats(self):
        self.mean_scores.clear()
        return

    def after_print_stats(self, frame, epoch_num, total_time):
        if self.consecutive_successes.current_size > 0:
            mean_con_successes = self.consecutive_successes.get_mean()
            self.writer.add_scalar('successes/consecutive_successes/mean', mean_con_successes, frame)
            self.writer.add_scalar('successes/consecutive_successes/iter', mean_con_successes, epoch_num)
            self.writer.add_scalar('successes/consecutive_successes/time', mean_con_successes, total_time)
        return


class DefaultRewardsShaper:
    def __init__(self, scale_value = 1, shift_value = 0, min_val=-np.inf, max_val=np.inf, is_torch=True):
        self.scale_value = scale_value
        self.shift_value = shift_value
        self.min_val = min_val
        self.max_val = max_val
        self.is_torch = is_torch

    def __call__(self, reward):
        
        reward = reward + self.shift_value
        reward = reward * self.scale_value
 
        if self.is_torch:
            import torch
            reward = torch.clamp(reward, self.min_val, self.max_val)
        else:
            reward = np.clip(reward, self.min_val, self.max_val)
        return reward


class PhysHOIModelBuilder:
    def __init__(self, network_builder):
        self.network_builder = network_builder

    def build(self, config):
        net = self.network_builder.build(None, **config)
        for name, _ in net.named_parameters():
            print(name)
        return PhysHOIModelBuilder.Network(net)

    class Network(ModelA2CContinuousLogStd.Network):
        def __init__(self, a2c_network):
            super().__init__(a2c_network)
            return

        def forward(self, input_dict):
            result = super().forward(input_dict)
            return result    


# Replace rlgames' torch_runner and factories
class Runner:
    def __init__(self, algo_observer=None):
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
        network_builder = physhoi_network_builder.PhysHOIBuilder()
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
        return physhoi_players.PhysHOIPlayerContinuous(self.config)

    def run_train(self):
        print('Started to train')
        if self.algo_observer is None:
            self.algo_observer = RLGPUAlgoObserver()

        self.reset()
        self.load_config(self.default_config)

        if 'features' not in self.config:
            self.config['features'] = {}
        self.config['features']['observer'] = self.algo_observer

        #if 'soft_augmentation' in self.config['features']:
        #    self.config['features']['soft_augmentation'] = SoftAugmentation(**self.config['features']['soft_augmentation'])

        agent = physhoi_agent.PhysHOIAgent(base_name='run', config=self.config)

        if self.load_check_point and (self.load_path is not None):
            agent.restore(self.load_path)

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
        args.motion_file = "physhoi/data/motions/BallPlay/toss.pt"
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

    # Create and register the env creator with args, cfg, cfg_train
    env_configurations.register('rlgpu', {
        'env_creator': lambda **kwargs: create_rlgpu_env(args, cfg, cfg_train, **kwargs),
        'vecenv_type': 'RLGPU'})

    runner = Runner()
    runner.load(cfg_train)
    runner.reset()

    runner.run(vargs)
