# in this file we are going to load the trained policy and check if it has learned the reaching task.

import numpy as np
import gym
import random
import torch
from collections import deque
import matplotlib.pyplot as plt
from copy import copy
from HER.DDPG_Agent import DDPG_Agent


# environment, I need some parameters from it to instantiate the Actor
ENV = 'FetchReach-v1'
# ENV = 'FetchPickAndPlace-v1'
# ENV = 'FetchPush-v1'
VERBOSE = False
ACTOR_LEARNING_RATE = 1e-3
CRITIC_LEARNING_RATE = 1e-3
BUFFER_CAPACITY = int(1e6)
BATCH_SIZE = 128
NOISE_TYPE = "Normal"
NOISE_STDDEV = 0.2
TAU = 0.05
GAMMA = 0.98
N_EPOCHS = 200
N_CYCLES = 50
N_EPISODES_PER_CYCLES = 16
N_OPTIMIZATION_STEPS = 40
N_REPLAYED_GOALS = 4
NORMALIZED_OBSERVATIONS = False

env = gym.make(ENV)
env.seed(100)
# env.env.distance_threshold = 0.02
print(env.env.initial_gripper_xpos[:3])
obs_shape = env.observation_space.spaces['observation'].shape[0]
action_shape = env.action_space.low.size
goal_shape = env.observation_space.spaces['desired_goal'].low.size
max_action_value = float(env.action_space.high[0])
min_action_value = float(env.action_space.low[0])

print('Information about the environment:')
print('...Observation dimension: ', obs_shape)
print('...Goal dimension: ', goal_shape)
print('...Action dimension: ', action_shape)
print('...Max value for the action:', max_action_value)
print('...Min value for the action:', min_action_value)
print('...Distance threshold:', env.env.distance_threshold)


# I guess I only need the critic, since we are interested in the policy and we are only evaluating it and not training
# anymore
print('\n...Loading trained policy')
agent = DDPG_Agent(env, obs_shape, goal_shape, action_shape, max_action_value, min_action_value,
                   replay_buffer_capacity=BUFFER_CAPACITY, actor_lr = ACTOR_LEARNING_RATE, critic_lr= CRITIC_LEARNING_RATE, batch_size=BATCH_SIZE, tau=TAU, gamma=GAMMA,
                   noise_stddev=NOISE_STDDEV, noise_type=NOISE_TYPE, normalized_observations=NORMALIZED_OBSERVATIONS)
agent.actor.load_state_dict(torch.load('../HER/checkpoints/checkpoint_actor_final_FetchReach-v1_env_Normal_noise_setting_hyper_Future_sampling_strategy_until_80_10_cycle_opt_every_3_512bs.pth'))
agent.critic.load_state_dict(torch.load('../HER/checkpoints/checkpoint_critic_final_FetchReach-v1_env_Normal_noise_setting_hyper_Future_sampling_strategy_until_80_10_cycle_opt_every_3_512bs.pth'))

N_TEST_EXAMPLE = 100
n_success = 0
done = False

with torch.no_grad():
    for _ in range(N_TEST_EXAMPLE):
        obs = env.reset()
        # here if we want we can try different goals written by us
        goal = obs['desired_goal']
        print(goal)
        done = False
        while not done:
            env.render()
            state = obs['observation']
            action = agent.select_action(state, goal, add_noise=False, use_target_network=False)
            next_obs, reward, done, _ = env.step(action)
            if reward == 0.0:
                done = True
                n_success += 1

            obs = next_obs

print('We were able to solve {} out of {} levels'.format(n_success, N_TEST_EXAMPLE))
# default_goals = np.array([[1.40403629, 0.41440915, 0.88495493],
#                  [1.52444277, 1.07594242, 0.80419867],
#                  [1.43756641, 1.05764926, 0.84741368],
#                  [1.52444277, 0.97937922, 0.73952689]])

# default_success = 0
# # print(n_success)
# for i in range(len(default_goals)):
#     obs = env.reset()
#     env.env.goal = default_goals[i]
#     obs['desired_goal'] = env.env.goal
#     goal = obs['desired_goal']
#     print(goal)
#     done = False
#
#     while not done:
#         env.render()
#         state = obs['observation']
#         action = agent.select_action(state, goal, add_noise=False)
#         next_obs, reward, done, info = env.step(action)
#         if reward == 0.0:
#             done = True
#             default_success += 1
#         obs = next_obs
#         assert np.array_equal(next_obs['desired_goal'], goal), "Problem with the goal assignment"