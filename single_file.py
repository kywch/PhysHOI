# python physhoi/run.py --test --task PhysHOI_BallPlay --num_envs 4 --cfg_env physhoi/data/cfg/physhoi.yaml --cfg_train physhoi/data/cfg/train/rlg/physhoi.yaml --motion_file physhoi/data/motions/BallPlay/toss.pt --checkpoint physhoi/data/models/toss/nn/PhysHOI.pth
from pdb import set_trace as T

from physhoi.utils.config import set_np_formatting, get_args, load_cfg, parse_sim_params
from env import RLGPUEnv, create_rlgpu_env

from rl_games.common import env_configurations, vecenv
from rl_games.algos_torch import torch_ext
from rl_games.common.algo_observer import AlgoObserver
from rl_games.torch_runner import Runner

from physhoi.learning import physhoi_agent
from physhoi.learning import physhoi_players
from physhoi.learning import physhoi_models
from physhoi.learning import physhoi_network_builder



set_np_formatting()
args = get_args()

run_eval = False

if run_eval:
    # CLI arguments for eval
    args.test = True
    args.task = "PhysHOI_BallPlay"
    args.num_envs = 4
    args.cfg_env = "physhoi/data/cfg/physhoi.yaml"
    args.cfg_train = "physhoi/data/cfg/train/rlg/physhoi.yaml"
    args.motion_file = "physhoi/data/motions/BallPlay/toss.pt"
    args.checkpoint = "physhoi/data/models/toss/nn/PhysHOI.pth"

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

# vec env?
vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
env_configurations.register('rlgpu', {
    'env_creator': lambda **kwargs: create_rlgpu_env(args, cfg, cfg_train, **kwargs),
    'vecenv_type': 'RLGPU'})

# rl-games
# class RLGPUAlgoObserver(AlgoObserver):
#     def __init__(self, use_successes=True):
#         self.use_successes = use_successes
#         return

#     def after_init(self, algo):
#         self.algo = algo
#         self.consecutive_successes = torch_ext.AverageMeter(1, self.algo.games_to_track).to(self.algo.ppo_device)
#         self.writer = self.algo.writer
#         return

#     def process_infos(self, infos, done_indices):
#         if isinstance(infos, dict):
#             if (self.use_successes == False) and 'consecutive_successes' in infos:
#                 cons_successes = infos['consecutive_successes'].clone()
#                 self.consecutive_successes.update(cons_successes.to(self.algo.ppo_device))
#             if self.use_successes and 'successes' in infos:
#                 successes = infos['successes'].clone()
#                 self.consecutive_successes.update(successes[done_indices].to(self.algo.ppo_device))
#         return

#     def after_clear_stats(self):
#         self.mean_scores.clear()
#         return

#     def after_print_stats(self, frame, epoch_num, total_time):
#         if self.consecutive_successes.current_size > 0:
#             mean_con_successes = self.consecutive_successes.get_mean()
#             self.writer.add_scalar('successes/consecutive_successes/mean', mean_con_successes, frame)
#             self.writer.add_scalar('successes/consecutive_successes/iter', mean_con_successes, epoch_num)
#             self.writer.add_scalar('successes/consecutive_successes/time', mean_con_successes, total_time)
#         return

# algo_observer = RLGPUAlgoObserver()

def build_alg_runner():
    runner = Runner()

    runner.algo_factory.register_builder('physhoi', lambda **kwargs : physhoi_agent.PhysHOIAgent(**kwargs))
    runner.player_factory.register_builder('physhoi', lambda **kwargs : physhoi_players.PhysHOIPlayerContinuous(**kwargs))
    runner.model_builder.model_factory.register_builder('physhoi', lambda network, **kwargs : physhoi_models.ModelPhysHOIContinuous(network))  
    runner.model_builder.network_factory.register_builder('physhoi', lambda **kwargs : physhoi_network_builder.PhysHOIBuilder())

    return runner

runner = build_alg_runner()
runner.load(cfg_train)
runner.reset()

runner.run(vargs)
