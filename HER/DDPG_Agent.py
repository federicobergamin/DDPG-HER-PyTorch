'''
Implementation of DDPG (Deep Deterministic Policy Gradient)
paper: https://arxiv.org/abs/1509.02971

From the paper we know that the DDPG is a model-free,
off-policy, actor-critic algorithm that uses a deep
function approximation.
'''

#TODO: we should modify this code in a way that deals with goals --> they clip also in the range of [-5,5]
from copy import copy, deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import HER.utils as utils
from HER.models import Actor, Critic
from HER.noise import Normal_Action_Noise, Ornstein_Uhlenbeck_Action_Noise
from HER.normalizer import Running_stats
from HER.replay_buffer import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------- Methods for normalizing and denormalizing ------------------
def normalize(observation, statistics):
    if statistics is None:
        return observation
    else:
        return (observation - statistics.running_mean) / statistics.running_stddev

def denormalize(observation, statistics):
    if statistics is None:
        return observation
    else:
        return observation * statistics.running_stddev + statistics.running_mean


# DDPG learns both a Q-function and a policy
# TODO: try to use the adaptive params noise instead of using noise to the actions
class DDPG_Agent(object):

    # maybe some params are useless --> we will see
    def __init__(self, env, observation_dim, goal_dim, action_dim, max_action, min_action, replay_buffer_capacity=int(1e6), actor_lr=1e-4, critic_lr=1e-3,
                 actor_weight_decay=0.00, critic_weight_decay=0.01, batch_size=64, tau=0.05, gamma=0.99, param_noise = False, action_noise = True,
                 noise_stddev=0.1, noise_type='Normal', normalized_observations = False, normalized_returns = False, noise_eps = 0.2, seed = 42):

        # np.random.seed(seed)
        # torch.manual_seed(seed)
        # variables initialization
        self.env = env

        # ---------- Initialize Actor-Critic ------------------ #
        self.observation_dim = observation_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.max_action_value = max_action
        self.min_action_value = min_action
        self.actor = Actor(self.observation_dim, self.goal_dim, self.action_dim, self.max_action_value, original_paper_parms_init=False).to(device)
        self.critic = Critic(self.observation_dim, self.goal_dim, self.action_dim, original_paper_parms_init=False).to(device)

        # ---------- Initialize Replay-Buffer ------------------ #
        self.replay_buffer_size = replay_buffer_capacity
        self.replay_buffer = ReplayBuffer(self.observation_dim, self.goal_dim, self.action_dim, self.replay_buffer_size)

        # ---------- Parameters for learning ------------------- #
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.batch_size = batch_size
        # self.epochs = epochs
        # self.steps_per_epoch = steps_per_epoch
        self.tau = tau
        self.gamma = gamma
        self.param_noise = param_noise
        self.action_noise = action_noise
        self.noise_stddev = noise_stddev
        self.noise_mean = np.zeros(self.action_dim)
        self.noise_type = noise_type
        self.normalized_observations = normalized_observations
        self.normalized_returns = normalized_returns
        self.seed = seed
        self.noise_eps = noise_eps

        # ---------- Networks' Optimizer  ------------------- #
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr, weight_decay=critic_weight_decay)

        # ---------- Target Networks ------------------- #
        self.actor_target = Actor(self.observation_dim, self.goal_dim, self.action_dim, self.max_action_value).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target = Critic(self.observation_dim, self.goal_dim, self.action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # ---------- Noise process ------------------- #
        if noise_type == 'Normal':
            self.noise = Normal_Action_Noise(self.noise_mean, self.noise_stddev * np.ones(self.action_dim))
        elif noise_type == 'OU':
            self.noise = Ornstein_Uhlenbeck_Action_Noise(self.action_dim, self.seed)
        else:
            raise RuntimeError('unknown noise type "{}"'.format(self.noise_type))

        # ---------- Running average and stddev for normalization ------------------- #
        # if normalized observation and/or actions is true, we
        # have to initialize a tracker for their statistics
        if self.normalized_observations:
            # in this case we want normalize both the observations and the goal since we have to concatenate them
            self.obs_stats = Running_stats(self.observation_dim)
            self.goal_stats = Running_stats(self.goal_dim)
        else:
            self.obs_stats = None
            self.goal_stats = None

        if self.normalized_returns:
            self.qval_stat = Running_stats(1) # qvalues are single numbers
        else:
            self.qval_stat = None


    ## function for the soft update of the parameters of the target network
    def soft_update(self, target_network, source_network):
        ''' Update the parameters of the target network in a soft way (using the self.tau param) '''
        for target_param, param in zip(target_network.parameters(), source_network.parameters()):
            # target_param.detach()
            # print(param.data)
            target_param.data.copy_(target_param.data * (1 - self.tau) + param.data * self.tau)


    ## getting the evaluation of the prediction in numpy (maybe useless)
    # def eval_step(self, state):
    #     state = normalize(state, self.obs_stats)
    #     action = self.actor(state)
    #     return action.cpu().detach().numpy()


    def select_action(self, state, goal, add_noise=True, use_target_network=False, already_normalized = False):
        ''' We use the actor network to get an action given the state following the policy.
            The add_noise is needed because we do not want to add noise at
            test time. (add_noise=True only for training)      '''
        # we normalize the observation
        # print('pre')
        # print(state)
        if self.normalized_observations and not already_normalized:
            state = normalize(state, self.obs_stats).clip(-5,5)
            goal = normalize(goal, self.goal_stats).clip(-5,5)

        # we can concatenate them
        input = np.hstack((state,goal))
        # print(input.shape)
        # state = torch.from_numpy(state).float().to(device)
        # goal = torch.from_numpy(goal).float().to(device)
        input = torch.from_numpy(input).float()
        # print('post')
        # print(state)
        if use_target_network:
            self.actor_target.eval()
            with torch.no_grad():
                action, _ = self.actor_target(input)
                action = action.cpu().data.numpy()
            # action = action * self.max_action_value
            self.actor_target.train()
        else:
            self.actor.eval()
            with torch.no_grad():
                action, _ = self.actor(input)
                action = action.cpu().data.numpy()
                # print(action)
                # action = action * self.max_action_value
            self.actor.train()
        if add_noise:
            # print(self.noise())
            # print(action)
            noise = self.noise.sample() * self.max_action_value * self.noise_eps
            # print(noise)
            action = np.add(action, noise)
            # print(action)
        # return action
        # print(self.min_action_value)
        # print(self.max_action_value)
        # print(torch.clamp(action, self.min_action_value, self.max_action_value))
        return np.clip(action, self.min_action_value, self.max_action_value)
        # return np.clip(action, -1, 1)

        # return action


    def store_transition(self, state, desired_goal, action, next_state, reward, done):
        ''' We insert a transition into the replay buffer and
            update the statistics of the observations (mean and stddev) '''

        if self.normalized_observations:
            state = normalize(state, self.obs_stats).clip(-5,5)
            next_state = normalize(next_state, self.obs_stats).clip(-5,5)
            desired_goal = normalize(desired_goal, self.goal_stats).clip(-5,5)
            # state = torch.from_numpy(state).float().to(device)
            # next_state = torch.from_numpy(next_state).float().to(device)

        # we can sotre also arrays maybe
        # obs_shape = state.shape[0]
        # print(obs_shape)
        # print(state)
        # for obs in range(obs_shape):
        #     self.replay_buffer.add_sample(state[obs], action[obs], next_state[obs], reward[obs], done[obs])
        #     if self.normalized_observations:
        #         with torch.no_grad():
        #             self.obs_stats.update(torch.Tensor([state[obs]]))
        # if self.normalized_observations:
        #     state= state.numpy()
        #     next_state = next_state.numpy()

        self.replay_buffer.add_sample(state, desired_goal, action, next_state, reward, done)
        if self.normalized_observations:
            with torch.no_grad():
                self.obs_stats.update_stats(state)
                self.goal_stats.update_stats(desired_goal)

    def store_trajectory(self, trajectory):
        ''' We are given a trajectory, and we have to add each transition of the trajectory into the replay buffer,
            and normalize it '''
        observations = trajectory['observations']
        actions = trajectory['actions']
        rewards = trajectory['rewards']
        next_observations = trajectory['next_observations']
        done = trajectory['done']
        desired_goals = trajectory['desired_goals']

        trajectory_len = len(observations)

        for i in range(trajectory_len):
            self.store_transition(observations[i], desired_goals[i], actions[i], next_observations[i], rewards[i], done[i])



    def learn(self):
        ''' We have to start with sampling from our replay buffer 'batch_size' transitions,
            for each transition we have to compute the Q_target    '''

        # we start with sampling from the buffer
        # experiences will be a dict
        experiences = self.replay_buffer.sample(self.batch_size)

        # we shall unpack the dictionary
        # and the state and next_state should be already normalize
        states = experiences['observations']
        desired_goals = experiences['desired_goals']
        actions = experiences['actions']
        rewards = experiences['rewards']
        next_states = experiences['next_observations']
        dones = experiences['done']

        # ------------- State-Goal Concatenation --------------
        # we have to concatenate the states and the goals
        observation = torch.from_numpy(np.hstack((states, desired_goals)))
        next_observation = torch.from_numpy(np.hstack((next_states, desired_goals)))

        # -------------- Critic Loss ------------
        # we have to compute the Q_target = reward[i] + gamma * critic_target(next_state, actor_target(next_state))
        next_actions, _ = self.actor_target(next_observation)
        # next_actions += self.noise.sample() * self.max_action_value * self.noise_eps
        # next_actions = torch.from_numpy(self.select_action(states,desired_goals, add_noise=True, use_target_network=True, already_normalized=True)).float().to(device)
        next_actions = next_actions.detach()

        next_Q_targets_values = self.critic_target(next_observation, next_actions)
        # Q_targets = rewards + (self.gamma * next_Q_targets_values * (1 - dones))
        Q_targets = rewards + (self.gamma * next_Q_targets_values)
        Q_targets = Q_targets.detach()
        # Here I am clipping the targets
        Q_targets = torch.clamp(Q_targets, -1/(1-self.gamma),0)

        # we also need the expected one for computing the loss
        Q_pred = self.critic(observation, actions)
        # compute the loss of the critic
        loss_func = nn.MSELoss()
        critic_loss = loss_func(Q_pred,Q_targets)
        # we have to backpropagate
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # -------------- Actor Loss ------------
        action_pred, preactivations = self.actor(observation)
        # actor_loss_preactivation = (preactivations**2).sum(dim=1).mean()
        # Q_value_pred = self.critic(states, action_pred)
        # actor_loss = -Q_value_pred.mean()
        # with torch.no_grad():
        self.critic.eval()
        actor_loss = - self.critic(observation, action_pred).mean()
        self.critic.train()
        # actor_loss = -1 * q_pred
        actor_loss += 1.0 * ((preactivations)**2).sum(dim=1).mean()
        # we want to minimize this loss --> in reality we want to maximixe the expected Q
        # actor_loss += actor_loss_preactivation
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ------------- Update both actor_target and critic_target ---------
        self.soft_update(self.critic_target, self.critic)
        self.soft_update(self.actor_target, self.actor)

    def reset(self):
        if self.noise_type == 'OU':
            self.noise.reset()




















































