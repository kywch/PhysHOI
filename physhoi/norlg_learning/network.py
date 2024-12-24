import numpy as np
import torch
import torch.nn as nn


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """CleanRL's default layer initialization"""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PhysHOINetworkBuilder:
    def __init__(self, **kwargs):
        self.params = None

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        assert self.params is not None, "params is not set"
        net = PhysHOINetworkBuilder.Network(self.params, **kwargs)
        return net

    class Network(nn.Module):
        def __init__(self, params, **kwargs):
            actions_num = kwargs.pop('actions_num')
            input_shape = kwargs.pop('input_shape')
            hidden1, hidden2 = 1024, 512

            super().__init__()

            # Replace self.load(params)
            self.is_continuous = True
            self.units = params['mlp']['units']
            self.space_config = params['space']['continuous']

            # Fix the network, to be the same as the original
            # mlp:
            #   units: [1024, 512]
            #   activation: relu
            input_size = input_shape[0]
            out_size = self.units[-1]

            # TODO: remove these
            self.actor_cnn = nn.Sequential()
            self.critic_cnn = nn.Sequential()

            # Separate actor and critic networks
            self.actor_mlp = nn.Sequential(
                layer_init(nn.Linear(input_size, hidden1)),
                nn.ReLU(),
                layer_init(nn.Linear(hidden1, hidden2)),
                nn.ReLU(),
            )

            self.mu = nn.Linear(out_size, actions_num)
            self.mu_act = nn.Identity()  # self.activations_factory.create(self.space_config['mu_activation']) 
            self.sigma_act = nn.Identity()  # self.activations_factory.create(self.space_config['sigma_activation']) 

            if self.space_config['learn_sigma']:
                self.sigma = nn.Linear(out_size, actions_num)
                nn.init.constant_(self.sigma.weight, self.space_config['sigma_init']['val'])
            else:
                self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=False, dtype=torch.float32), requires_grad=False)
                nn.init.constant_(self.sigma, self.space_config['sigma_init']['val'])

            # Critic network
            self.critic_mlp = nn.Sequential(
                layer_init(nn.Linear(input_size, hidden1)),
                nn.ReLU(),
                layer_init(nn.Linear(hidden1, hidden2)),
                nn.ReLU(),
            )

            self.value = nn.Linear(out_size, 1)
            self.value_act = nn.ReLU()  # self.activations_factory.create(self.value_activation)

        def forward(self, obs_dict):
            obs = obs_dict['obs']
            states = obs_dict.get('rnn_states', None)

            actor_outputs = self.eval_actor(obs)
            value = self.eval_critic(obs)

            output = actor_outputs + (value, states)

            return output

        def eval_actor(self, obs):
            a_out = self.actor_cnn(obs)
            a_out = a_out.contiguous().view(a_out.size(0), -1)
            a_out = self.actor_mlp(a_out)
                     
            # if self.is_discrete:
            #     logits = self.logits(a_out)
            #     return logits

            # if self.is_multi_discrete:
            #     logits = [logit(a_out) for logit in self.logits]
            #     return logits

            if self.is_continuous:
                mu = self.mu_act(self.mu(a_out))
                if self.space_config['learn_sigma']:
                    sigma = self.sigma_act(self.sigma(a_out))
                else:
                    sigma = mu * 0.0 + self.sigma_act(self.sigma)
                return mu, sigma
            return

        def eval_critic(self, obs):
            c_out = self.critic_cnn(obs)
            c_out = c_out.contiguous().view(c_out.size(0), -1)
            c_out = self.critic_mlp(c_out)              
            value = self.value_act(self.value(c_out))
            return value


class PhysHOIModelBuilder:
    def __init__(self, network_builder):
        self.network_builder = network_builder

    def build(self, config):
        net = self.network_builder.build(None, **config)
        for name, _ in net.named_parameters():
            print(name)
        return PhysHOIModelBuilder.Network(net)

    # from rl_games.algos_torch.models import ModelA2CContinuousLogStd
    # class Network(ModelA2CContinuousLogStd.Network):
    class Network(nn.Module):
        def __init__(self, a2c_network):
            nn.Module.__init__(self)
            self.a2c_network = a2c_network

        def is_rnn(self):
            return False  # self.a2c_network.is_rnn()
            
        def get_default_rnn_state(self):
            return None  # self.a2c_network.get_default_rnn_state()

        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            prev_actions = input_dict.get('prev_actions', None)
            mu, logstd, value, states = self.a2c_network(input_dict)
            sigma = torch.exp(logstd)
            distr = torch.distributions.Normal(mu, sigma)
            if is_train:
                entropy = distr.entropy().sum(dim=-1)
                prev_neglogp = self.neglogp(prev_actions, mu, sigma, logstd)
                result = {
                    'prev_neglogp' : torch.squeeze(prev_neglogp),
                    'values' : value,
                    'entropy' : entropy,
                    'rnn_states' : states,
                    'mus' : mu,
                    'sigmas' : sigma
                }                
                return result
            else:
                selected_action = distr.sample()
                neglogp = self.neglogp(selected_action, mu, sigma, logstd)
                result = {
                    'neglogpacs' : torch.squeeze(neglogp),
                    'values' : value,
                    'actions' : selected_action,
                    'rnn_states' : states,
                    'mus' : mu,
                    'sigmas' : sigma
                }
                return result

        def neglogp(self, x, mean, std, logstd):
            return 0.5 * (((x - mean) / std)**2).sum(dim=-1) \
                + 0.5 * np.log(2.0 * np.pi) * x.size()[-1] \
                + logstd.sum(dim=-1)
