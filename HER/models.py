## actor - critic

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from HER.utils import hidden_layer_init, output_layer_init

# np.random.seed(42)
# torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    ''' Actor Network using the parameters suggested by the paper: https://arxiv.org/pdf/1707.01495.pdf'''

    def __init__(self, observation_dim, goal_dim, action_dim, max_action, original_paper_parms_init=False):
        super(Actor, self).__init__()

        # 3 fully connected layers with 64 hidden units each
        # self.begin = nn.LayerNorm(observation_dim + goal_dim)
        self.fc1 = nn.Linear(observation_dim + goal_dim, 64)
        self.fc1_norm = nn.LayerNorm(64)
        self.fc2 = nn.Linear(64, 64)
        self.fc2_norm = nn.LayerNorm(64)
        self.fc3 = nn.Linear(64, action_dim)
        # self.fc3_norm = nn.LayerNorm(64)
        # self.fc4 = nn.Linear(64, action_dim)

        self.max_action = max_action

        if original_paper_parms_init:
            hidden_layer_init(self.fc1)
            hidden_layer_init(self.fc2)
            output_layer_init(self.fc3)


    def forward(self, x):
        # print(x)
        # hidden layer with ReLu activation
        x = F.relu(self.fc1_norm(self.fc1(x)))
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc1_norm(self.fc1(self.begin(x))))
        x = F.relu(self.fc2_norm(self.fc2(x)))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))

        preactivations = self.fc3(x)
        # print('preactivation')
        # print(preactivations)


        # output layer with tanh and rescale by the max_action
        return torch.tanh(preactivations) * self.max_action, preactivations


class Critic(nn.Module):
    ''' Critic Network using the parameters suggested by the paper '''

    def __init__(self, observation_dim, goal_dim, action_dim, original_paper_parms_init=False):
        super(Critic, self).__init__()

        # 3 fully connected layers with 64 hidden units each
        # self.action_norm = nn.LayerNorm(action_dim)
        # self.begin = nn.LayerNorm(observation_dim + goal_dim + action_dim)
        self.fc1 = nn.Linear(observation_dim + goal_dim + action_dim, 64)
        self.fc1_norm = nn.LayerNorm(64)
        # self.action_norm = nn.LayerNorm(action_dim)
        self.fc2 = nn.Linear(64, 64)
        self.fc2_norm = nn.LayerNorm(64)
        self.fc3 = nn.Linear(64, 1)
        # self.fc3_norm = nn.LayerNorm(64)
        # self.fc4 = nn.Linear(64, 1)

        if original_paper_parms_init:
            hidden_layer_init(self.fc1)
            hidden_layer_init(self.fc2)
            output_layer_init(self.fc3)

    def forward(self, x, actor_output):
        # actor_output = self.action_norm(actor_output)
        x = F.relu(self.fc1_norm(self.fc1(torch.cat((x,actor_output),1))))
        # x = F.relu(self.fc1(torch.cat((x,actor_output),1)))
        # x = F.relu(self.fc1_norm(self.fc1(self.begin(torch.cat((x,actor_output),1)))))
        # x = torch.cat((x,self.action_norm(actor_output)),1)
        x = F.relu(self.fc2_norm(self.fc2(x)))
        # x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        # q_value = self.fc4(x)
        return q_value






