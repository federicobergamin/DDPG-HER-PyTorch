''' For now I will use this file to understand how to use Fetch Environment
    and the goal-oriented task such as reach, push, sliding, where we ask for
    a desired goal, but at the same time, for each state we obtained an achieved_goal
'''

import numpy as np
import h5py
import gym
from HER.her_goals_sampling_strategies import final_sampling_strategy, future_sampling_strategies
from HER.trajectory import Trajectory
from HER.utils import sample_new_goal


# now I need a simple sampling strategy for the goals up and down
SEED = 1000
X_RANGE = [1.05,1.45]
Y_RANGE = [0.45,1.00]
Z_RANGE = [0.37+(0.89-0.37)/2,0.78]
down = ((0.89-0.37)/2.0)
print(down)
### we will sample from 0.37 + 0.26 and then the rest
np.random.seed(SEED)
##------------ DETERMINISTIC Valid SET GOAL -----------------
N_VALID_EXAMPLE = 100
validation_goal = []
for _ in range(N_VALID_EXAMPLE):
    validation_goal.append([np.random.uniform(X_RANGE[0],X_RANGE[1]), np.random.uniform(Y_RANGE[0],Y_RANGE[1]), np.random.uniform(Z_RANGE[0],Z_RANGE[1])])


validation_goal = np.array(validation_goal)
print(validation_goal)

# store the performances
file_path_test = "../HER/deterministic_test_sets/default_bottom_test_set_created_with_seed_{}".format(SEED)

file_5 = h5py.File(file_path_test, 'w')

file_5.create_dataset('test_set', data=validation_goal, chunks = True, compression = "gzip")

file_5.close()

# env = gym.make('FetchPush-v1')
# # initial_qpos = {'robot0:grip': np.array([1.45, 1.04, 0.8])}
# # env.env.sim.__init__(initial_qpos=initial_qpos)
#
# print('Information about the environment')
# print('Observation dimension: ',env.observation_space.spaces['observation'].low.size)
# print('Goal dimension: ', env.observation_space.spaces['desired_goal'].low.size)
# print('Action dimension: ', env.action_space.low.size)
# print(env.observation_space.spaces['info'])
# print(env.env.target_offset)
# print(env.env.target_range)
# print('----- Gripper Xpos --------')
# print(env.env.initial_gripper_xpos)
# print(env.env.distance_threshold)
# env.env.goal = np.array([0.5,0.5,0.5])
# obs = env.reset()
# print(obs)
#
#
# for _ in range(50):
#     obs = env.reset()
#     env.render()
#
#     print(obs['desired_goal'])
#
#
#

# print(obs['desired_goal'])
# orig = obs['desired_goal']

# (largehzza, lunghezza, altezza)
## TODO: NOTICE: to change the position of the goal or of the robot
# env.env.goal = np.array([1.05,0.40,0.9])
# env.env.distance_threshold = 0.01
# obs['desired_goal'] = env.env.goal
# env.env.initial_gripper_xpos = np.array([1.45, 1.04, 0.8])
# env.env.sim.initial_gripper_xpos['robot0:grip'] = np.array([1.45, 1.04, 0.8])
# print('------------')
# print(env.env.sim.data.get_site_xpos('robot0:grip'))
# print('-------------')
# print(obs)

# for _ in range(6):
#     obs = env.reset()
#     sample_new_goal(env, obs, x_range=[0.1,0.2], y_range=[10,11], z_range=[4,5])
#     print(obs['desired_goal'])
#


# default_goals = [[1.40403629, 0.41440915, 0.88495493],
#                  [1.52444277, 1.07594242, 0.80419867],
#                  [1.43756641, 1.05764926, 0.84741368]]
#
# default_success = 0
# # print(n_success)
# for i in range(len(default_goals)):
#     obs = env.reset()
#     env.env.goal = default_goals[i]
#     obs['desired_goal'] = env.env.goal
#     goal = obs['desired_goal']
#     done = False
#
#     while not done:
#         state = obs['observation']
#         action = obs.select_action(state, goal, add_noise=False)
#         next_obs, reward, done, info = env.step(action)
#         if reward == 0.0:
#             done = True
#             default_success += 1
#         obs = next_obs
#         assert np.array_equal(next_obs['desired_goal'], goal), "Problem with the goal assignment"

# test_goals = []
# np.random.seed(42)
# for _ in range(10):
#     test_goals.append([np.random.uniform(1.05,1.55), np.random.uniform(0.4,1.10), np.random.uniform(0.4,0.9)])
#
# test_goals = np.array(test_goals)
# print(test_goals)
#
# for i in range(len(test_goals)):
#     obs = env.reset()
#     env.env.goal = test_goals[i]
#     env.env.distance_threshold = 0.001
#     obs['desired_goal'] = env.env.goal
#     print('--------')
#     print(obs)
#     print('--------')
#
# done = False
#
# def policy(observation, desired_goal):
#     # Here you would implement your smarter policy. In this case,
#     # we just sample random actions.
#     return env.action_space.sample()
#
# trajectory = Trajectory()
# iter = 0
# done_ = False
# while not done_:
#     iter +=1
#     print('Iteration n.{}'.format(iter))
#     # env.goal = np.array([0.5,0.5,0.5])
#     action = policy(obs['observation'], obs['desired_goal'])
#     # print(action)
#     next_obs, reward, done, info = env.step(action)
#
#     print('CHECK THE REWARD IF WE GET RIGHT')
#     assert reward == env.compute_reward(next_obs['achieved_goal'], next_obs['desired_goal'], info)
#
#     substitute_goal = next_obs['achieved_goal'].copy()
#     substitute_reward = env.compute_reward(next_obs['achieved_goal'], substitute_goal, info)
#     print(substitute_reward)
#     # print(reward)
#     # print(done)
#     # print(info)
#     # print('----')
#     if reward == 0.0:
#         print('UNITED WE WIN')
#         # print('%%%%%%%%%%%%%')
#         # print(obs['desired_goal'])
#         # print(obs['achieved_goal'])
#     # print(obs['desired_goal'])
#     # print(next_obs['desired_goal'])
#     assert np.array_equal(obs['desired_goal'], next_obs['desired_goal'])
#     trajectory.add_transition(observations=obs['observation'],
#                               actions=action,
#                               next_observations = next_obs['observation'],
#                               desired_goals = obs['desired_goal'],
#                               rewards=reward,
#                               achieved_goals=next_obs['achieved_goal'],
#                               infos = info)
#
#
#     obs = next_obs
#
#
#
# ## at this point we have the final trajectory of 10 steps
# # print('Safety check')
# # print(trajectory.get_complete_trajectory())
#
# print('------------------')
# print('------------------')
# print('------------------')
# print('------------------')
#
# # print('Sampling strategy final')
# # sampled_trajectory = final_sampling_strategy(env, trajectory)
# # print(sampled_trajectory.get_complete_trajectory())
# # TODO --> try to store the achieve goal to check if the reward is right
# print('Sampling strategy future')
# sampled_trajectory = future_sampling_strategies(env, trajectory, 2)
# tr = sampled_trajectory.get_complete_trajectory()
# print('--------------')
# print('desired goal: ', tr['desired_goals'][1])
# print('achieved_goal: ', tr['achieved_goals'][1])
# print('reward: ', tr['rewards'][1])
# print('--------------')
# print('desired goal: ',tr['desired_goals'][4])
# print('achieved_goal: ', tr['achieved_goals'][4])
# print('reward: ',tr['rewards'][4])
# print('--------------')
# print('desired goal: ',tr['desired_goals'][3])
# print('achieved_goal: ', tr['achieved_goals'][3])
# print('reward: ',tr['rewards'][3])
# print('--------------')
# print('desired goal: ',tr['desired_goals'][5])
# print('achieved_goal: ', tr['achieved_goals'][5])
# print('reward: ',tr['rewards'][5])
#




















    #############
    ##initial example only to check the environment
    ##
    #############
    # print('-----')
    # print(reward)
    # print('-----')
    # print(info)
    # print(obs['desired_goal'])
    # if done:
    #     print('Iterations ', iter)
    #     print('checking the threshold')
    #     print(obs['desired_goal'])
    #     print(obs['achieved_goal'])
    #     d = np.linalg.norm(obs['desired_goal'] - obs['achieved_goal'], axis=-1)
    #     print(d)
    #     d2 = np.linalg.norm(orig - obs['achieved_goal'], axis=-1)
    #     print(d2)
    # print(obs['achieved_goal'])
    # env.render()
    #
    #
    # If we want, we can substitute a goal here and re-compute
    # the reward. For instance, we can just pretend that the desired
    # goal was what we achieved all along.
    # substitute_goal = obs['achieved_goal'].copy()
    # substitute_reward = env.compute_reward(
    #     obs['achieved_goal'], substitute_goal, info=None)
    # print('reward is {}, substitute_reward is {}'.format(
    #     reward, substitute_reward))

