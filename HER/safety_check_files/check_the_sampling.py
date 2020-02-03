import numpy as np
import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from copy import copy
from HER.her_goals_sampling_strategies import final_sampling_strategy, future_sampling_strategies
from HER.trajectory import Trajectory
from HER.DDPG_Agent import DDPG_Agent
from HER.trajectory import Trajectory

def future_sampling_strategy(reward_func, trajectory, n_sampled_goal):
    observations = trajectory['observations']
    actions = trajectory['actions']
    rewards = trajectory['rewards']
    next_observations = trajectory['next_observations']
    desired_goals = trajectory['desired_goals']
    achieved_goals = trajectory['achieved_goals']
    info = trajectory['info']

    sampled_transitions = Trajectory()
    T = len(observations)

    # in this case we follow the future strategy
    for i in range(T):
        for _ in range(n_sampled_goal):
            # we have to sample a goal from i to the end
            # i guess that when we reach end-i<n_sampled_goal,
            # we sample that one? or what? we sample n_sampled_goal time the same thing? #TODO: TRY WHAT CHANGES
            if i == len(observations):
                sampled_goal_idx = i
            else:
                sampled_goal_idx = np.random.randint(i, len(observations))

            # now we have to recompute the reward
            new_reward = reward_func(achieved_goals[i], achieved_goals[sampled_goal_idx])

            sampled_transitions.add_transition(observations=observations[i],
                                               actions=actions[i],
                                               desired_goals=achieved_goals[sampled_goal_idx],
                                               rewards=new_reward,
                                               next_observations=next_observations[i],
                                               info=info[i],
                                               achieved_goals=achieved_goals[i])
    return sampled_transitions


def compute_reward(goal_1, goal_2):
    return 0 if goal_1==goal_2 else -1

def final_sampling_strategy(reward, trajectory):
    # we get the initial trajectory as parameter
    observations = trajectory['observations']
    actions = trajectory['actions']
    rewards = trajectory['rewards']
    next_observations = trajectory['next_observations']
    desired_goals = trajectory['desired_goals']
    achieved_goals = trajectory['achieved_goals']
    info = trajectory['info']

    sampled_transitions = Trajectory()
    # for each transition in the trajectory, we have to sample the final episode achieved goal
    # and use that one as desired goal for all transitions --> we also have to recompute the goal
    # final goal achieved in the final state
    final_goal_episode = achieved_goals[-1]
    # print(final_goal_episode)
    for i in range(len(observations)):
        # we have to recompute the reward before being able to add the transition
        new_reward = reward(achieved_goals[i], final_goal_episode)
        sampled_transitions.add_transition(observations=observations[i],
                                           actions=actions[i],
                                           desired_goals=final_goal_episode,
                                           rewards=new_reward,
                                           next_observations=next_observations[i],
                                           info=info[i])

    return sampled_transitions


traj = dict(
    observations = [1,2,3,4,5,6,7,8,9,10],
    actions = [0,0,0,0,0,0,0,0,0,0],
    rewards = [1,1,1,1,1,1,1,1,1,1],
    next_observations = [2,3,4,5,6,7,8,9,10,11],
    desired_goals = [50,50,50,50,50,50,50,50,50,50],
    achieved_goals = [2,3,4,5,6,7,8,9,10,11],
    info = ['boh','boh','boh','boh','boh','boh','boh','boh','boh','boh'])

sampled_traj = future_sampling_strategy(compute_reward, traj,2)
print(sampled_traj)

