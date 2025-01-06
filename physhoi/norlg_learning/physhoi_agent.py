import os
import time
import shutil
from datetime import datetime

import torch 
from torch import optim

import physhoi.norlg_learning.physhoi_players as physhoi_players
from physhoi.norlg_learning.amp_datasets import AMPDataset
from physhoi.norlg_learning.utils import AverageMeter, ExperienceBuffer

from tensorboardX import SummaryWriter


def to_torch(x, dtype=torch.float, device='cuda:0', requires_grad=False):
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)

def mean_list(val):
    return torch.mean(torch.stack(val))

def swap_and_flatten01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    if arr is None:
        return arr
    s = arr.size()
    return arr.transpose(0, 1).reshape(s[0] * s[1], *s[2:])

def policy_kl(p0_mu, p0_sigma, p1_mu, p1_sigma, reduce=True):
    c1 = torch.log(p1_sigma/p0_sigma + 1e-5)
    c2 = (p0_sigma**2 + (p1_mu - p0_mu)**2)/(2.0 * (p1_sigma**2 + 1e-5))
    c3 = -1.0 / 2.0
    kl = c1 + c2 + c3
    kl = kl.sum(dim=-1) # returning mean between all steps of sum between all actions
    if reduce:
        return kl.mean()
    else:
        return kl


class PhysHOIAgent(physhoi_players.PhysHOIAgent):
    def __init__(self, config, env):
        super().__init__(config, env)

        assert self._amp_input_mean_std is not None, "ampinput_mean_std must be set"

	    # Train-specific
        self.use_action_masks = False  # config.get('use_action_masks', False)
        self.is_train = True  # config.get('is_train', True)
        self.ppo = True  # config['ppo']
        self.save_freq = config.get('save_frequency', 0)
        self.max_epochs = config.get('max_epochs', 0)

        self.num_actors = config['num_actors']
        self.num_agents = self.env_info.get('agents', 1)

        self.algo_observer = config.get('algo_observer', None)
        self.network = config['network']
        self.rewards_shaper = config['reward_shaper']  # CHECK ME

        self.network_path = config.get('network_path', "./nn/")
        self.log_path = config.get('log_path', "runs/")

        # PPO-related
        self.e_clip = config['e_clip']
        self.clip_value = config['clip_value']
        self.horizon_length = config['horizon_length']
        self.normalize_advantage = config['normalize_advantage']
        self.normalize_input = config['normalize_input']
        self.critic_coef = config['critic_coef']
        self.grad_norm = config['grad_norm']
        self.gamma = config['gamma']
        self.tau = config['tau']
        self.entropy_coef = config['entropy_coef']
        self.bounds_loss_coef = config.get('bounds_loss_coef', None)

        # self.games_num = self.config['minibatch_size'] // self.seq_len # it is used only for current rnn implementation
        self.batch_size = self.horizon_length * self.num_actors * self.num_agents
        self.batch_size_envs = self.horizon_length * self.num_actors
        self.minibatch_size = config['minibatch_size']
        self.mini_epochs_num = config['mini_epochs']
        self.num_minibatches = self.batch_size // self.minibatch_size
        assert(self.batch_size % self.minibatch_size == 0)

        self.obs = None
        self.last_lr = float(config['learning_rate'])
        self.frame = 0
        self.update_time = 0
        self.mean_rewards = self.last_mean_rewards = -100500
        self.play_time = 0
        self.epoch_num = 0
        self.curr_frames = 0

        self.optimizer = optim.Adam(self.model.parameters(), self.last_lr, eps=1e-08, weight_decay=0.0)

        self.dataset = AMPDataset(self.batch_size, self.minibatch_size, self.device)

        self.resume_from = config['resume_from']
        self.done_indices = []

        # CHECK ME
        # self.truncate_grads = self.config.get('truncate_grads', False)
        # self.has_phasic_policy_gradients = False
        # self.normalize_rms_advantage = False  # config.get('normalize_rms_advantage', False)
        # self.value_bootstrap = self.config.get('value_bootstrap')

        # remove?
        self.mixed_precision = False  # self.config.get('mixed_precision', False)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

        self.games_to_track = self.config.get('games_to_track', 100)
        self.game_rewards = AverageMeter(self.value_size, self.games_to_track).to(self.device)
        self.game_lengths = AverageMeter(1, self.games_to_track).to(self.device)

        # Check folders
        # allows us to specify a folder where all experiments will reside
        self.train_dir = config.get('train_dir', 'runs')

        # a folder inside of train_dir containing everything related to a particular experiment
        self.experiment_name = config['name'] + datetime.now().strftime("_%m-%d-%H-%M-%S")
        self.experiment_dir = os.path.join(self.train_dir, self.experiment_name)

        # folders inside <train_dir>/<experiment_dir> for a specific purpose
        self.nn_dir = os.path.join(self.experiment_dir, 'nn')
        self.summaries_dir = os.path.join(self.experiment_dir, 'summaries')

        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.summaries_dir, exist_ok=True)

        self.writer = SummaryWriter(self.summaries_dir)
        self.algo_observer.set_writer(self.writer)
        self.algo_observer.set_consecutive_successes(
            AverageMeter(1, self.games_to_track).to(self.device)
        )

        print()

    # CHECK ME: is it really unwrapped?
    @property
    def unwrapped_env(self):
        return self.env.env

    def init_tensors(self):
        batch_size = self.num_agents * self.num_actors
        algo_info = {
            'num_actors' : self.num_actors,
            'horizon_length' : self.horizon_length,
        }
        self.experience_buffer = ExperienceBuffer(self.env_info, algo_info, self.device)

        current_rewards_shape = (batch_size, self.value_size)
        self.current_rewards = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.device)
        self.current_lengths = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        self.dones = torch.ones((batch_size,), dtype=torch.uint8, device=self.device)

        self.experience_buffer.tensor_dict['next_obses'] = torch.zeros_like(self.experience_buffer.tensor_dict['obses'])
        self.experience_buffer.tensor_dict['next_values'] = torch.zeros_like(self.experience_buffer.tensor_dict['values'])

        self.update_list = ['actions', 'neglogpacs', 'values', 'mus', 'sigmas']
        self.tensor_list = self.update_list + ['obses', 'states', 'dones', 'next_obses', 'amp_obs']
    
    def save(self, file_path):
        print("=> saving checkpoint '{}'".format(file_path))
        state_dict = self.get_model_weights()

        # Training state
        state_dict['epoch'] = self.epoch_num
        state_dict['optimizer'] = self.optimizer.state_dict()
        state_dict['frame'] = self.frame

        # This is actually the best reward ever achieved. last_mean_rewards is perhaps not the best variable name
        # We save it to the checkpoint to prevent overriding the "best ever" checkpoint upon experiment restart
        state_dict['last_mean_rewards'] = self.last_mean_rewards

        # env_state = self.unwrapped_env.get_env_state()
        # state_dict['env_state'] = env_state

        # Save the checkpoint
        torch.save(state_dict, file_path)

    def train(self):
        # CHECK ME: is resume_from necessary?
        # if self.resume_from != 'None':
        #     self.restore(self.resume_from)
        self.init_tensors()
        self.last_mean_rewards = -100500
        total_time = 0
        self.frame = 0  # frame = agent step

        self.obs = self.env_reset()
        self.curr_frames = self.batch_size_envs
        
        model_output_file = os.path.join(self.nn_dir, self.config['name'])

        epoch_num = 0
        while True:
            epoch_num += 1
            self.epoch_num = epoch_num

            # collect data
            start_time = time.time()
            with torch.no_grad():
                batch_dict = self.play_steps()
            scaled_play_time = time.time() - start_time

            # update the model
            update_time_start = time.time()
            train_info = None

            self.prepare_dataset(batch_dict)
            for _ in range(0, self.mini_epochs_num):
                for i in range(len(self.dataset)):
                    curr_train_info = self.calc_gradients(self.dataset[i])  # updating

                    if (train_info is None):
                        train_info = dict()
                        for k, v in curr_train_info.items():
                            train_info[k] = [v]
                    else:
                        for k, v in curr_train_info.items():
                            train_info[k].append(v)

            train_info['play_time'] = scaled_play_time
            train_info['update_time'] = time.time() - update_time_start
            sum_time = time.time() - start_time
            total_time += sum_time
            frame = self.frame

            # log the stats
            scaled_time = sum_time
            curr_frames = self.curr_frames
            self.frame += curr_frames
            if self.print_stats:
                fps_step = curr_frames / scaled_play_time
                fps_total = curr_frames / scaled_time
                print("epoch_num:{}".format(epoch_num), "mean_rewards:{}".format(self.game_rewards.get_mean()), f'fps step: {fps_step:.1f} fps total: {fps_total:.1f}')

            self.writer.add_scalar('performance/total_fps', curr_frames / scaled_time, frame)
            self.writer.add_scalar('performance/step_fps', curr_frames / scaled_play_time, frame)
            self.writer.add_scalar('info/epochs', epoch_num, frame)
            self._log_train_info(train_info, frame)

            self.algo_observer.after_print_stats(frame, epoch_num, total_time)
            
            # save the checkpoint
            if self.save_freq > 0:
                if (epoch_num % self.save_freq == 0):
                    self.save(model_output_file)

                    # save the intermediate checkpoints
                    int_model_output_file = model_output_file + '_' + str(epoch_num).zfill(8)
                    shutil.copyfile(model_output_file, int_model_output_file)

            if epoch_num > self.max_epochs:
                self.save(model_output_file)
                print('Reached the maximum number of epochs. Finshed training.')
                return self.last_mean_rewards, epoch_num

    def get_action_values(self, obs_torch):
        processed_obs = self._preproc_obs(obs_torch)

        self.model.eval()
        with torch.no_grad():
            res_dict = self.model({
                'is_train': False,
                'prev_actions': None, 
                'obs' : processed_obs,
            })

        return res_dict

    def _eval_critic(self, obs_torch):
        self.model.eval()
        processed_obs = self._preproc_obs(obs_torch)  # normalize the obs
        return self.model.a2c_network.eval_critic(processed_obs)

    def play_steps(self):
        self.set_eval()
        update_list = self.update_list

        for n in range(self.horizon_length):
            self.obs = self.env_reset(self.done_indices)
            self.experience_buffer.update_data('obses', n, self.obs)

            res_dict = self.get_action_values(self.obs)
            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k]) 

            """Stepping the environment"""
            # self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
            self.obs, rewards, self.dones, infos = self.env.step(res_dict['actions'])

            if self.value_size == 1:
                rewards = rewards.unsqueeze(1)

            # No special reward shaping used. Remove.
            # shaped_rewards = self.rewards_shaper(rewards)
            shaped_rewards = rewards  # shape error

            self.experience_buffer.update_data('rewards', n, shaped_rewards)
            self.experience_buffer.update_data('next_obses', n, self.obs)
            self.experience_buffer.update_data('dones', n, self.dones)
            # self.experience_buffer.update_data('rand_action_mask', n, res_dict['rand_action_mask'])

            terminated = infos['terminate'].float()
            terminated = terminated.unsqueeze(-1)
            next_vals = self._eval_critic(self.obs)
            next_vals *= (1.0 - terminated)
            self.experience_buffer.update_data('next_values', n, next_vals)

            self.current_rewards += rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            self.done_indices = all_done_indices[::self.num_agents]
  
            self.game_rewards.update(self.current_rewards[self.done_indices])
            self.game_lengths.update(self.current_lengths[self.done_indices])
            self.algo_observer.process_infos(infos, self.done_indices)

            not_dones = 1.0 - self.dones.float()
            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones
            
            if (self.unwrapped_env.task.viewer):
                self._amp_debug(infos)
                
            self.done_indices = self.done_indices[:, 0]

        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_next_values = self.experience_buffer.tensor_dict['next_values']
        mb_rewards = self.experience_buffer.tensor_dict['rewards']

        # Calculate the advantages
        lastgaelam = 0
        mb_advs = torch.zeros_like(mb_rewards)
        for t in reversed(range(self.horizon_length)):
            not_done = 1.0 - mb_fdones[t]
            not_done = not_done.unsqueeze(1)
            delta = mb_rewards[t] + self.gamma * mb_next_values[t] - mb_values[t]
            lastgaelam = delta + self.gamma * self.tau * not_done * lastgaelam
            mb_advs[t] = lastgaelam
        mb_returns = mb_advs + mb_values

        # Return the batch data for training
        batch_dict = self.experience_buffer.get_transformed_list(swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size

        return batch_dict

    def prepare_dataset(self, batch_dict):
        advantages = batch_dict['returns'] - batch_dict['values']
        advantages = torch.sum(advantages, axis=1)
        if self.normalize_advantage:
            adv_mean = torch.mean(advantages)
            adv_std = torch.std(advantages)
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)

        dataset_dict = {}
        dataset_dict['old_values'] = batch_dict['values']
        dataset_dict['old_logp_actions'] = batch_dict['neglogpacs']
        dataset_dict['advantages'] = advantages
        dataset_dict['returns'] = batch_dict['returns']
        dataset_dict['actions'] = batch_dict['actions']
        dataset_dict['obs'] = batch_dict['obses']
        dataset_dict['mu'] = batch_dict['mus']
        dataset_dict['sigma'] = batch_dict['sigmas']

        self.dataset.update_values_dict(dataset_dict)

    def _clip_policy_loss(self, old_action_log_probs_batch, action_log_probs, advantage, curr_e_clip):
        # clipping the policy loss
        ratio = torch.exp(old_action_log_probs_batch - action_log_probs)
        surr1 = advantage * ratio
        surr2 = advantage * torch.clamp(ratio, 1.0 - curr_e_clip,
                                    1.0 + curr_e_clip)
        a_loss = torch.max(-surr1, -surr2)
        clipped = torch.abs(ratio - 1.0) > curr_e_clip
        return {
            'actor_loss': a_loss,
            'actor_clipped': clipped.detach()
        }

    def _clip_value_loss(self, value_preds_batch, values, curr_e_clip, return_batch, clip_value):
        # clipping the value loss
        if clip_value:
            value_pred_clipped = value_preds_batch + \
                    (values - value_preds_batch).clamp(-curr_e_clip, curr_e_clip)
            value_losses = (values - return_batch)**2
            value_losses_clipped = (value_pred_clipped - return_batch)**2
            c_loss = torch.max(value_losses, value_losses_clipped)
        else:
            c_loss = (return_batch - values)**2

        return {
            'critic_loss': c_loss
        }

    def bound_loss(self, mu):
        if self.bounds_loss_coef is not None:
            soft_bound = 1.0
            mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0)**2
            mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0)**2
            b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
        else:
            b_loss = 0
        return b_loss

    def calc_gradients(self, input_dict):
        self.set_train()

        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']
        obs_batch = self._preproc_obs(obs_batch)
        
        lr_mul = 1.0
        curr_e_clip = lr_mul * self.e_clip

        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch, 
            'obs' : obs_batch
        }

        with torch.cuda.amp.autocast(enabled=False):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']

            a_info = self._clip_policy_loss(old_action_log_probs_batch, action_log_probs, advantage, curr_e_clip)
            a_loss = a_info['actor_loss']
            a_clipped = a_info['actor_clipped'].float()

            c_info = self._clip_value_loss(value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
            c_loss = c_info['critic_loss']

            b_loss = self.bound_loss(mu)

            c_loss = torch.mean(c_loss)
            a_loss = torch.mean(a_loss) 
            entropy = torch.mean(entropy)
            b_loss = torch.mean(b_loss)
            a_clip_frac = torch.mean(a_clipped)
            
            loss = a_loss + self.critic_coef * c_loss + self.bounds_loss_coef * b_loss
            
            a_info['actor_loss'] = a_loss
            a_info['actor_clip_frac'] = a_clip_frac
            c_info['critic_loss'] = c_loss

            self.optimizer.zero_grad()

        # TODO: remove self.scaler
        # self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        with torch.no_grad():
            kl_dist = policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch)
                    
        self.train_result = {
            'entropy': entropy,
            'kl': kl_dist,
            'last_lr': self.last_lr, 
            'lr_mul': lr_mul, 
            'b_loss': b_loss
        }
        self.train_result.update(a_info)
        self.train_result.update(c_info)

        return self.train_result

    def _log_train_info(self, train_info, frame):
        self.writer.add_scalar('performance/update_time', train_info['update_time'], frame)
        self.writer.add_scalar('performance/play_time', train_info['play_time'], frame)
        self.writer.add_scalar('losses/a_loss', mean_list(train_info['actor_loss']).item(), frame)
        self.writer.add_scalar('losses/c_loss', mean_list(train_info['critic_loss']).item(), frame)
        
        self.writer.add_scalar('losses/bounds_loss', mean_list(train_info['b_loss']).item(), frame)
        self.writer.add_scalar('losses/entropy', mean_list(train_info['entropy']).item(), frame)
        self.writer.add_scalar('info/last_lr', train_info['last_lr'][-1] * train_info['lr_mul'][-1], frame)
        self.writer.add_scalar('info/lr_mul', train_info['lr_mul'][-1], frame)
        self.writer.add_scalar('info/e_clip', self.e_clip * train_info['lr_mul'][-1], frame)
        self.writer.add_scalar('info/clip_frac', mean_list(train_info['actor_clip_frac']).item(), frame)
        self.writer.add_scalar('info/kl', mean_list(train_info['kl']).item(), frame)
        return

    def _amp_debug(self, info):
        return
