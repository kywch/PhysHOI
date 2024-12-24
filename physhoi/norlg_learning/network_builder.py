import numpy as np
import torch
import torch.nn as nn

# from rl_games.algos_torch.network_builder import NetworkBuilder

from physhoi.norlg_learning.utils import shape_whc_to_cwh


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """CleanRL's default layer initialization"""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class A2CNetworkBuilder:  # (NetworkBuilder):
    def __init__(self, **kwargs):
        self.params = None

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        assert self.params is not None, "params is not set"
        net = A2CNetworkBuilder.Network(self.params, **kwargs)
        return net

    class SimpleNetwork(nn.Module):
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

            # if self.space_config['fixed_sigma']:
            #     self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)
            #     nn.init.constant_(self.sigma, self.space_config['sigma_init']['val'])
            # else:
            #     self.sigma = nn.Linear(out_size, actions_num)
            #     nn.init.constant_(self.sigma.weight, self.space_config['sigma_init']['val'])

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
            raise NotImplementedError


    class Network(nn.Module):  #(NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            actions_num = kwargs.pop('actions_num')
            input_shape = kwargs.pop('input_shape')
            self.value_size = kwargs.pop('value_size', 1)
            self.num_seqs = kwargs.pop('num_seqs', 1)

            # NetworkBuilder.BaseNetwork.__init__(self)
            nn.Module.__init__(self, **kwargs)
            self.load(params)
            self.actor_cnn = nn.Sequential()
            self.critic_cnn = nn.Sequential()
            self.actor_mlp = nn.Sequential()
            self.critic_mlp = nn.Sequential()
            
            # if self.has_cnn:
            #     input_shape = shape_whc_to_cwh(input_shape)
            #     cnn_args = {
            #         'ctype' : self.cnn['type'], 
            #         'input_shape' : input_shape, 
            #         'convs' :self.cnn['convs'], 
            #         'activation' : self.cnn['activation'], 
            #         'norm_func_name' : self.normalization,
            #     }
            #     self.actor_cnn = self._build_conv(**cnn_args)

            #     if self.separate:
            #         self.critic_cnn = self._build_conv( **cnn_args)

            mlp_input_shape = self._calc_input_size(input_shape, self.actor_cnn)

            in_mlp_shape = mlp_input_shape
            if len(self.units) == 0:
                out_size = mlp_input_shape
            else:
                out_size = self.units[-1]

            # if self.has_rnn:
            #     if not self.is_rnn_before_mlp:
            #         rnn_in_size = out_size
            #         out_size = self.rnn_units
            #         if self.rnn_concat_input:
            #             rnn_in_size += in_mlp_shape
            #     else:
            #         rnn_in_size =  in_mlp_shape
            #         in_mlp_shape = self.rnn_units

            #     if self.separate:
            #         self.a_rnn = self._build_rnn(self.rnn_name, rnn_in_size, self.rnn_units, self.rnn_layers)
            #         self.c_rnn = self._build_rnn(self.rnn_name, rnn_in_size, self.rnn_units, self.rnn_layers)
            #         if self.rnn_ln:
            #             self.a_layer_norm = nn.LayerNorm(self.rnn_units)
            #             self.c_layer_norm = nn.LayerNorm(self.rnn_units)
            #     else:
            #         self.rnn = self._build_rnn(self.rnn_name, rnn_in_size, self.rnn_units, self.rnn_layers)
            #         if self.rnn_ln:
            #             self.layer_norm = nn.LayerNorm(self.rnn_units)

            mlp_args = {
                'input_size' : in_mlp_shape, 
                'units' : self.units, 
                # 'activation' : self.activation, 
                # 'dense_func' : nn.Linear,
                # 'd2rl' : self.is_d2rl,
                'norm_func_name' : self.normalization,
                'norm_only_first_layer' : self.norm_only_first_layer
            }
            self.actor_mlp = self._build_sequential_mlp(**mlp_args)
            if self.separate:
                self.critic_mlp = self._build_sequential_mlp(**mlp_args)

            # Adding the action/value head to the network
            self.value = nn.Linear(out_size, self.value_size)
            self.value_act = nn.ReLU()  # self.activations_factory.create(self.value_activation)

            # if self.is_discrete:
            #     self.logits = nn.Linear(out_size, actions_num)
            # '''
            #     for multidiscrete actions num is a tuple
            # '''
            # if self.is_multi_discrete:
            #     self.logits = nn.ModuleList([nn.Linear(out_size, num) for num in actions_num])

            if self.is_continuous:
                self.mu = nn.Linear(out_size, actions_num)
                self.mu_act = nn.Identity()  # self.activations_factory.create(self.space_config['mu_activation']) 
                self.sigma_act = nn.Identity()  # self.activations_factory.create(self.space_config['sigma_activation']) 

                if self.space_config['fixed_sigma']:
                    self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)
                else:
                    self.sigma = nn.Linear(out_size, actions_num)

            mlp_init = nn.Identity()  # self.init_factory.create(**self.initializer)
            # if self.has_cnn:
            #     cnn_init = self.init_factory.create(**self.cnn['initializer'])

            for m in self.modules():         
                # if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                #     cnn_init(m.weight)
                #     if getattr(m, "bias", None) is not None:
                #         nn.init.zeros_(m.bias)
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        nn.init.zeros_(m.bias)

            if self.is_continuous:
                mu_init = nn.Identity()  # self.init_factory.create(**self.space_config['mu_init'])
                mu_init(self.mu.weight)

                # sigma_init = self.init_factory.create(**self.space_config['sigma_init'])  # nn.init.constant_
                if self.space_config['fixed_sigma']:
                    # sigma_init(self.sigma)
                    nn.init.constant_(self.sigma, self.space_config['sigma_init']['val'])
                else:
                    # sigma_init(self.sigma.weight)  
                    nn.init.constant_(self.sigma.weight, self.space_config['sigma_init']['val'])

        def forward(self, obs_dict):
            raise NotImplementedError

            # obs = obs_dict['obs']
            # states = obs_dict.get('rnn_states', None)
            # seq_length = obs_dict.get('seq_length', 1)
            # if self.has_cnn:
            #     # for obs shape 4
            #     # input expected shape (B, W, H, C)
            #     # convert to (B, C, W, H)
            #     if len(obs.shape) == 4:
            #         obs = obs.permute((0, 3, 1, 2))

            # if self.separate:
            #     a_out = c_out = obs
            #     a_out = self.actor_cnn(a_out)
            #     a_out = a_out.contiguous().view(a_out.size(0), -1)

            #     c_out = self.critic_cnn(c_out)
            #     c_out = c_out.contiguous().view(c_out.size(0), -1)                    

            #     if self.has_rnn:
            #         if not self.is_rnn_before_mlp:
            #             a_out_in = a_out
            #             c_out_in = c_out
            #             a_out = self.actor_mlp(a_out_in)
            #             c_out = self.critic_mlp(c_out_in)

            #             if self.rnn_concat_input:
            #                 a_out = torch.cat([a_out, a_out_in], dim=1)
            #                 c_out = torch.cat([c_out, c_out_in], dim=1)

            #         batch_size = a_out.size()[0]
            #         num_seqs = batch_size // seq_length
            #         a_out = a_out.reshape(num_seqs, seq_length, -1)
            #         c_out = c_out.reshape(num_seqs, seq_length, -1)

            #         if self.rnn_name == 'sru':
            #             a_out =a_out.transpose(0,1)
            #             c_out =c_out.transpose(0,1)

            #         if len(states) == 2:
            #             a_states = states[0]
            #             c_states = states[1]
            #         else:
            #             a_states = states[:2]
            #             c_states = states[2:]                        
            #         a_out, a_states = self.a_rnn(a_out, a_states)
            #         c_out, c_states = self.c_rnn(c_out, c_states)
         
            #         if self.rnn_name == 'sru':
            #             a_out = a_out.transpose(0,1)
            #             c_out = c_out.transpose(0,1)
            #         else:
            #             if self.rnn_ln:
            #                 a_out = self.a_layer_norm(a_out)
            #                 c_out = self.c_layer_norm(c_out)
            #         a_out = a_out.contiguous().reshape(a_out.size()[0] * a_out.size()[1], -1)
            #         c_out = c_out.contiguous().reshape(c_out.size()[0] * c_out.size()[1], -1)

            #         if type(a_states) is not tuple:
            #             a_states = (a_states,)
            #             c_states = (c_states,)
            #         states = a_states + c_states

            #         if self.is_rnn_before_mlp:
            #             a_out = self.actor_mlp(a_out)
            #             c_out = self.critic_mlp(c_out)
            #     else:
            #         a_out = self.actor_mlp(a_out)
            #         c_out = self.critic_mlp(c_out)
                            
            #     value = self.value_act(self.value(c_out))

            #     if self.is_discrete:
            #         logits = self.logits(a_out)
            #         return logits, value, states

            #     if self.is_multi_discrete:
            #         logits = [logit(a_out) for logit in self.logits]
            #         return logits, value, states

            #     if self.is_continuous:
            #         mu = self.mu_act(self.mu(a_out))
            #         if self.space_config['fixed_sigma']:
            #             sigma = mu * 0.0 + self.sigma_act(self.sigma)
            #         else:
            #             sigma = self.sigma_act(self.sigma(a_out))

            #         return mu, sigma, value, states
            # else:
            #     out = obs
            #     out = self.actor_cnn(out)
            #     out = out.flatten(1)                

            #     if self.has_rnn:
            #         out_in = out
            #         if not self.is_rnn_before_mlp:
            #             out_in = out
            #             out = self.actor_mlp(out)
            #             if self.rnn_concat_input:
            #                 out = torch.cat([out, out_in], dim=1)

            #         batch_size = out.size()[0]
            #         num_seqs = batch_size // seq_length
            #         out = out.reshape(num_seqs, seq_length, -1)

            #         if len(states) == 1:
            #             states = states[0]

            #         if self.rnn_name == 'sru':
            #             out = out.transpose(0,1)

            #         out, states = self.rnn(out, states)
            #         out = out.contiguous().reshape(out.size()[0] * out.size()[1], -1)

            #         if self.rnn_name == 'sru':
            #             out = out.transpose(0,1)
            #         if self.rnn_ln:
            #             out = self.layer_norm(out)
            #         if self.is_rnn_before_mlp:
            #             out = self.actor_mlp(out)
            #         if type(states) is not tuple:
            #             states = (states,)
            #     else:
            #         out = self.actor_mlp(out)
            #     value = self.value_act(self.value(out))

            #     if self.central_value:
            #         return value, states

            #     if self.is_discrete:
            #         logits = self.logits(out)
            #         return logits, value, states
            #     if self.is_multi_discrete:
            #         logits = [logit(out) for logit in self.logits]
            #         return logits, value, states
            #     if self.is_continuous:
            #         mu = self.mu_act(self.mu(out))
            #         if self.space_config['fixed_sigma']:
            #             sigma = self.sigma_act(self.sigma)
            #         else:
            #             sigma = self.sigma_act(self.sigma(out))
            #         return mu, mu*0 + sigma, value, states
                    
        def is_separate_critic(self):
            return self.separate

        def is_rnn(self):
            return self.has_rnn

        # def get_default_rnn_state(self):
        #     if not self.has_rnn:
        #         return None
        #     num_layers = self.rnn_layers
        #     if self.rnn_name == 'identity':
        #         rnn_units = 1
        #     else:
        #         rnn_units = self.rnn_units
        #     if self.rnn_name == 'lstm':
        #         if self.separate:
        #             return (torch.zeros((num_layers, self.num_seqs, rnn_units)), 
        #                     torch.zeros((num_layers, self.num_seqs, rnn_units)),
        #                     torch.zeros((num_layers, self.num_seqs, rnn_units)), 
        #                     torch.zeros((num_layers, self.num_seqs, rnn_units)))
        #         else:
        #             return (torch.zeros((num_layers, self.num_seqs, rnn_units)), 
        #                     torch.zeros((num_layers, self.num_seqs, rnn_units)))
        #     else:
        #         if self.separate:
        #             return (torch.zeros((num_layers, self.num_seqs, rnn_units)), 
        #                     torch.zeros((num_layers, self.num_seqs, rnn_units)))
        #         else:
        #             return (torch.zeros((num_layers, self.num_seqs, rnn_units)),)                

        def load(self, params):
            self.separate = params.get('separate', False)  # true
            self.units = params['mlp']['units']
            self.activation = params['mlp']['activation']  # relu
            self.initializer = params['mlp']['initializer']  # default

            # NOTE: these don't seem necessary
            self.is_d2rl = params['mlp'].get('d2rl', False)  # false
            self.norm_only_first_layer = params['mlp'].get('norm_only_first_layer', False)  # none
            self.value_activation = params.get('value_activation', 'None')  # none
            self.normalization = params.get('normalization', None)  # none
            # self.has_space = 'space' in params  # true
            self.central_value = params.get('central_value', False)  # false
            self.joint_obs_actions_config = params.get('joint_obs_actions', None)  # none

            # physhoi is continuous
            assert 'space' in params, "space is not set"
            assert 'continuous' in params['space'], "continuous is not set"
            self.is_discrete = False
            self.is_continuous = True
            self.is_multi_discrete = False
            self.space_config = params['space']['continuous']

            # No RNN in physhoi
            self.has_rnn = False  # 'rnn' in params
            # if self.has_rnn:
            #     self.rnn_units = params['rnn']['units']
            #     self.rnn_layers = params['rnn']['layers']
            #     self.rnn_name = params['rnn']['name']
            #     self.rnn_ln = params['rnn'].get('layer_norm', False)
            #     self.is_rnn_before_mlp = params['rnn'].get('before_mlp', False)
            #     self.rnn_concat_input = params['rnn'].get('concat_input', False)

            # No CNN in physhoi
            self.has_cnn = False
            # if 'cnn' in params:
            #     self.has_cnn = True
            #     self.cnn = params['cnn']
            # else:
            #     self.has_cnn = False

        def _calc_input_size(self, input_shape, cnn_layers=None):
            if cnn_layers is None:
                assert(len(input_shape) == 1)
                return input_shape[0]
            else:
                return nn.Sequential(*cnn_layers)(torch.rand(1, *(input_shape))).flatten(1).data.size(1)

        def _build_sequential_mlp(self, 
            input_size, 
            units, 
            norm_only_first_layer=False, 
            norm_func_name = None
        ):
            print('build mlp:', input_size)
            
            # Fix the network, to be the same as the original
            # mlp:
            #   units: [1024, 512]
            #   activation: relu
            return nn.Sequential(
                nn.Linear(input_size, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
            )

            # NOTE: consider adding a layernorm
            # Learn to walk in 20 min: https://arxiv.org/abs/2208.07860
            # Used LayerNorm to regularize the critic

            # in_size = input_size
            # layers = []
            # need_norm = True
            # for unit in units:
            #     # layers.append(dense_func(in_size, unit))  # dense_func = nn.Linear
            #     # layers.append(self.activations_factory.create(activation))  # activation = relu
            #     layers.append(nn.Linear(in_size, unit))
            #     layers.append(nn.ReLU())

            #     if norm_only_first_layer and norm_func_name is not None:
            #        need_norm = False 
            #     if not need_norm:
            #         continue

            #     if norm_func_name == 'layer_norm':
            #         layers.append(nn.LayerNorm(unit))
            #     elif norm_func_name == 'batch_norm':
            #         layers.append(nn.BatchNorm1d(unit))

            #     in_size = unit

            # return nn.Sequential(*layers)

