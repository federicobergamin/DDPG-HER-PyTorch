'''
Implementation of the Acrtor and Critic networks
'''

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import DDPG_example.utils as utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# TODO: maybe the initialization of the hidden layer is useless because it is the same as the default one
class Actor(nn.Module):
    ''' Actor Network '''

    def __init__(self, obseravation_dim, action_dim, max_action):
        super(Actor, self).__init__()

        # self.normalized_input = nn.BatchNorm1d(obseravation_dim)
        self.fc1 = nn.Linear(obseravation_dim, 400)
        # self.fc1 = nn.Linear(obseravation_dim, 400)
        utils.hidden_layer_init(self.fc1)
        self.fc2 = nn.Linear(400, 300)
        utils.hidden_layer_init(self.fc2)
        self.fc3 = nn.Linear(300, action_dim)
        utils.output_layer_init(self.fc3)

        self.max_action = max_action


    def forward(self, x):
        # x = self.normalized_input(x)
        # print(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # action = torch.tanh(self.fc3(x))
        # print('%%%%%%%%%%%%%%%%%%%%%')
        # print(torch.tanh(self.fc3(x)))
        # print('pre')
        # print(torch.tanh(self.fc3(x)))
        # action = self.max_action * torch.tanh(self.fc3(x))
        # print('post')
        # print(action)
        action = torch.tanh(self.fc3(x)) * self.max_action
        # print('----------------')
        # print(action.shape)
        # print(action)
        return action

class Critic(nn.Module):
    ''' Critic Network
        It takes the action given by the actor
        only from the second layer (as stated in the paper '''

    def __init__(self, observation_dim, action_dim):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(observation_dim, 400)
        utils.hidden_layer_init(self.fc1)
        self.fc2 = nn.Linear(400 + action_dim, 300)
        utils.hidden_layer_init(self.fc2)
        self.fc3 = nn.Linear(300, 1)
        utils.output_layer_init(self.fc3)

    def forward(self, x, actor_output):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(torch.cat((x,actor_output),1)))
        value_state = self.fc3(x)
        return value_state









