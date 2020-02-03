## replay buffer for HER algorithm

import numpy as np
import torch
import torch.nn as nn
import random
from collections import namedtuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# main class
class ReplayBuffer(object):

    # constructor
    def __init__(self, observation_dim, goal_dim, action_dim, max_capacity):
        self.observation_dim = observation_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self._max_capacity = max_capacity

        # sets that store the transitions (s,a,r,s',done)
        self._observations = np.zeros((self._max_capacity, self.observation_dim))
        self._desired_goals = np.zeros((self._max_capacity, self.goal_dim))
        self._actions = np.zeros((self._max_capacity, self.action_dim))
        self._rewards = np.zeros((self._max_capacity))
        self._next_observations = np.zeros((self._max_capacity, self.observation_dim))
        self._done = np.zeros((self._max_capacity), dtype='uint8')
        # maybe  useless --> for training the network we need only the desired one
        self._achieved_goals = np.zeros((self._max_capacity, self.goal_dim))

        ## this is useful only for add_trajectories --> if we change the way we sample, we should consider to optimise
        # this part of code
        self._indices_for_future_sampling_strategy = [None] * self._max_capacity

        self._size = 0
        self._position = 0

    def add_sample(self, observation, desired_goal, action, next_observation, reward, done, **kwargs):
        ## when the buffer is full oldest samples are discarded
        self._observations[self._position] = observation
        self._desired_goals[self._position] = desired_goal
        self._actions[self._position] = action
        self._next_observations[self._position] = next_observation
        self._rewards[self._position] = reward
        self._done[self._position] = done
        # self._achieved_goals[self._position] = achieved_goal
        self.update_indices()

    ## maybe we can continue with using add_sample if we follow the storing and sampling procedure described
    ## by algorithm 1 in the paper https://arxiv.org/pdf/1707.01495.pdf
    # we need the statistics, to normalize both the
    def add_trajectory(self, trajectory, normalized_observation_bool, obs_stats, goal_stats):

        # we have to retrieve all the parts of a trajectory (s,a,r,s',d,g)
        # (observations, actions, rewards, next_observations, dones, goals)
        observations = trajectory['observations']
        actions = trajectory['actions']
        rewards = trajectory['rewards']
        next_observations = trajectory['next_observations']
        done = trajectory['done']
        desired_goals = trajectory['desired_goals']
        # achieved_goals = trajectory['achieved_goals']

        trajectory_len = len(observations)

        # ----------------- normalization procedure ----------------
        if normalized_observation_bool:
            scaled_observations = []
            scaled_goals = []
            scaled_next_obs = []
            for i in range(trajectory_len):
                norm_obs = normalize(observations[i], obs_stats).clip(-5,5)
                scaled_observations.append(norm_obs)
                # print(next_observations[i])
                scaled_next_obs.append(normalize(next_observations[i], obs_stats).clip(-5,5))
                norm_goal = normalize(desired_goals[i], goal_stats).clip(-5,5)
                scaled_goals.append(norm_goal)
                ## update statistics
                with torch.no_grad():
                    obs_stats.update_stats(norm_obs)
                    goal_stats.update_stats(norm_goal)

            # transform them in numpy array
            observations = np.array(scaled_observations)
            desired_goals = np.array(scaled_goals)
            next_observations = np.array(scaled_next_obs)

        # now we have to check how many transitions we can store from the point we are,
        # to the end of the buffer

        # easy case: from the point we are there are trajectory_len spaces
        if self._position + trajectory_len <= self._max_capacity:
            self._observations[self._position:self._position+trajectory_len] = observations
            self._actions[self._position:self._position + trajectory_len] = actions
            self._rewards[self._position:self._position + trajectory_len] = rewards
            self._next_observations[self._position:self._position + trajectory_len] = next_observations
            self._done[self._position:self._position + trajectory_len] = done
            self._desired_goals[self._position:self._position + trajectory_len] = desired_goals
            # self._achieved_goals[self._position:self._position + trajectory_len] = achieved_goals

            # for each transition, we have to store the indices of the valid goal we can smaple
            # for i in range(self._position, self._position + trajectory_len):
            #     self._indices_for_future_sampling_strategy[i] = np.arange(i, self._position + trajectory_len)

        else:
            # more interesting case in which we are not able to fit all transitions without having an overflow
            # number of transitions we can fit before the end
            n_transitions_before_the_end = self._max_capacity - self._position
            n_transitions_overflow = trajectory_len - n_transitions_before_the_end

            # TODO: find a more elegant implementation (maybe)

            # -------------- PRE-OVERFLOW ------------
            assert len(self._observations[self._position:]) == len(observations[0:n_transitions_before_the_end])
            self._observations[self._position:] = observations[0:n_transitions_before_the_end]
            self._actions[self._position:] = actions[0:n_transitions_before_the_end]
            self._rewards[self._position:] = rewards[0:n_transitions_before_the_end]
            self._next_observations[self._position:] = next_observations[0:n_transitions_before_the_end]
            self._done[self._position:] = done[0:n_transitions_before_the_end]
            self._desired_goals[self._position:] = desired_goals[0:n_transitions_before_the_end]
            # self._achieved_goals[self._position:] = achieved_goals[0:n_transitions_before_the_end]

            # -------------- AFTER-OVERFLOW ------------
            assert len(self._observations[0:n_transitions_overflow]) == len(observations[n_transitions_before_the_end:])
            self._observations[0:n_transitions_overflow] = observations[n_transitions_before_the_end:]
            self._actions[0:n_transitions_overflow] = actions[n_transitions_before_the_end:]
            self._rewards[0:n_transitions_overflow] = rewards[n_transitions_before_the_end:]
            self._next_observations[0:n_transitions_overflow] = next_observations[n_transitions_before_the_end:]
            self._done[0:n_transitions_overflow] = done[n_transitions_before_the_end:]
            self._desired_goals[0:n_transitions_overflow] = desired_goals[n_transitions_before_the_end:]
            # self._achieved_goals[0:n_transitions_overflow] = achieved_goals[n_transitions_before_the_end:]

            # in this case also keeping the valid indices for each transitions for the future sampling is less straight
            # forward
            # indices for the transitions stored before the end of the buffer (we have to take into account also the
            # transitions that are at the beginning of the buffer
            # for i in range(self._position, self._position+ n_transitions_before_the_end):
            #     self._indices_for_future_sampling_strategy[i] = np.hstack((np.arange(i, self._max_capacity), np.arange(0,n_transitions_overflow)))
            #
            # # indices for the transition that overflow and that goes at the beginning of the buffer
            # for i in range(0, n_transitions_overflow):
            #     self._indices_for_future_sampling_strategy[i] = np.arange(i, n_transitions_overflow)

        ### in both cases we have to update the indices
        self._position = (self._position + trajectory_len) % self._max_capacity
        self._size = np.minimum(self._position + trajectory_len, self._max_capacity)


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
            desired_goals= torch.from_numpy(self._desired_goals[sampled_indices]).float().to(device),
            actions = torch.from_numpy(self._actions[sampled_indices]).float().to(device),
            rewards = torch.from_numpy(self._rewards[sampled_indices]).float().view(batch_size, -1).to(device),
            done = torch.from_numpy(self._done[sampled_indices]).float().view(batch_size, -1).to(device),
            next_observations = torch.from_numpy(self._next_observations[sampled_indices]).float().to(device)
        )

    def size(self):
        # print(self._size)
        # print(len(self._next_observations))
        # assert self._size == len(self._next_observations)
        return self._size

