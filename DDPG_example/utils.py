import numpy as np
import torch
import torch.nn as nn
import random
from collections import namedtuple

# we have to create the replay buffer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# TODO: maybe use namedtuple --> make the sample easier
#Transition = namedtuple('Transition',
#                       ('state','action','next_state','reward'))

class ReplayBuffer(object):
    '''We are going to store tuples (initial_state, action, next_state, reward, done)
       We store them in a different way instead, since in this way we avoid a for loop
       during the sample
    '''

    def __init__(self, max_capacity, observation_dim, action_dim):
        # instead of using a general memory, we could already divide it into different parts
        #self.memory = []
        self._obeservation_dim = observation_dim
        self._action_dim = action_dim
        # different dictionaries
        self._observations = np.zeros((max_capacity, observation_dim))
        self._actions = np.zeros((max_capacity, action_dim))
        self._next_observations = np.zeros((max_capacity, observation_dim))
        self._rewards = np.zeros((max_capacity,1))
        # the done we get
        self._done = np.zeros((max_capacity,1), dtype='uint8')

        self._max_capacity = max_capacity
        self._size = 0
        self._position = 0

    def add_sample(self, observation, action, next_observation, reward, done, **kwargs):
        ## when the buffer is full oldest samples are discarded
        self._observations[self._position] = observation
        self._actions[self._position] = action
        self._next_observations[self._position] = next_observation
        self._rewards[self._position] = reward
        self._done[self._position] = done
        self.update_indices()

    def update_indices(self):
        self._position = (self._position + 1) % self._max_capacity
        # print(self._position)
        if self._size < self._max_capacity:
            # print(self._size)
            # we should also update the real size --> most of the time we operate with the full buffer
            self._size += 1

    def sample(self, batch_size):
        sampled_indices = np.random.randint(0, self._size, batch_size)
        # print(len(sampled_indices))
        return dict(
            observations = torch.from_numpy(self._observations[sampled_indices]).float().to(device),
            actions = torch.from_numpy(self._actions[sampled_indices]).float().to(device),
            rewards = torch.from_numpy(self._rewards[sampled_indices]).float().view(batch_size,-1).to(device),
            done = torch.from_numpy(self._done[sampled_indices]).float().view(batch_size,-1).to(device),
            next_observations = torch.from_numpy(self._next_observations[sampled_indices]).float().to(device)
        )

    def size(self):
        # print(self._size)
        # print(len(self._next_observations))
        # assert self._size == len(self._next_observations)
        return self._size


def hidden_layer_init(hidden_layer):
    ''' We follow the initialization of the original paper:
        both weights and bias are initialized using a uniform distribution
        [-1/np.sqrt(f),1/np.sqrt(f)] where f is the fan_in of the layer '''
    # we start computing the fan in
    fan_in = hidden_layer.weight.data.size()[0]
    # fan_in = torch.nn.init._calculate_correct_fan(hidden_layer.weight, mode='fan_in')
    # torch.nn.init.uniform_(hidden_layer.weight, - 1. / np.sqrt(fan_in), 1. / np.sqrt(fan_in))
    # torch.nn.init.uniform_(hidden_layer.bias, -1. / np.sqrt(fan_in), 1. / np.sqrt(fan_in))
    lim = 1. / np.sqrt(fan_in)
    hidden_layer.weight.data.uniform_(-lim, lim)
    # hidden_layer.bias.data.uniform_(-lim, 1. /lim)


def output_layer_init(output_layer):
    ''' We use the same initialization as in the original paper'''
    # torch.nn.init.uniform_(output_layer.weight, -3 * 10e-4, 3 * 10e-4)
    # torch.nn.init.uniform_(output_layer.bias, -3 * 10e-4, 3 * 10e-4)
    lim= 3e-3
    output_layer.weight.data.uniform_(-lim, lim)
    output_layer.bias.data.uniform_(-lim, lim)











