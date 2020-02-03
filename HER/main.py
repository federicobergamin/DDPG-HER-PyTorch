
import gym
import random
import torch
import _pickle as pkl
import h5py
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from copy import copy
from HER.her_goals_sampling_strategies import final_sampling_strategy, future_sampling_strategies
from HER.trajectory import Trajectory
from HER.DDPG_Agent import DDPG_Agent
from HER.utils import sample_new_goal
# from gym.spaces.prng import seed

HER_SAMPLING = True
VERBOSE = False
# SEED = 1
# training_infos = 'reach_up_up_task_space_seed_{}'.format(SEED)

##### SAMPLING GOAL FUNCTION RANGE
# X_RANGE = [1.05,1.55]
# Y_RANGE = [0.4,1.10]
# Z_RANGE = [0.37,0.9]
# X_RANGE = [1.05,1.50]
# Y_RANGE = [0.4,1.00]
Z_RANGE = [0.39,0.37+(0.89-0.37)/2]
X_RANGE = [1.05,1.55]
Y_RANGE = [0.4,1.10]
# Z_RANGE = [0.37+(0.89-0.37)/2,0.85]
print(0.37+(0.89-0.37)/2)
### we will sample from 0.37 + 0.26 and then the rest
np.random.seed(42)
##------------ DETERMINISTIC Valid SET GOAL -----------------
N_VALID_EXAMPLE = 40
validation_goal = []
for _ in range(N_VALID_EXAMPLE):
    validation_goal.append([np.random.uniform(1.14, 1.42), np.random.uniform(0.5, 0.9), np.random.uniform(Z_RANGE[0],Z_RANGE[1])])

# print('\n Validation goal')
validation_goal = np.array(validation_goal)
# print(validation_goal)


# SEED = 42

##------------ DETERMINISTIC TEST SET GOAL -----------------
# we have to create ten goals randomly, selecting a seed to having them always the same
N_TEST_EXAMPLES = 40
test_goals = []
## set the seed
# torch.manual_seed(SEED)
# random.seed(SEED)
# np.random.seed(SEED)
# np.random.seed(70)
for _ in range(N_TEST_EXAMPLES):
    #     test_goals.append([np.random.uniform(1.05,1.55), np.random.uniform(0.4,1.10), np.random.uniform(0.4,0.9)])
    test_goals.append([np.random.uniform(1.05,1.50), np.random.uniform(0.4,1.00), np.random.uniform(Z_RANGE[0],Z_RANGE[1])])
    # test_goals.append([np.random.uniform(1.1,1.5), np.random.uniform(0.5,0.9), np.random.uniform(0.4,0.8)])
    # test_goals.append([np.random.uniform(1.1, 1.5), np.random.uniform(0.5, 1.0), np.random.uniform(0.5, 0.8)])
# print('Test goal')
test_goals = np.array(test_goals)
# print(test_goals)

filename = "../HER/deterministic_test_sets/default_bottom_test_set_created_with_seed_1000"
f = h5py.File(filename, 'r')
default_goals = np.array(f['test_set'])

for i in [666]:
    SEED = i
    training_infos = 'reach_bottom_policy_seed'.format(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    # deterministic test set
    # filename = "../HER/deterministic_test_sets/default_up_test_set_created_with_seed_1000"
    # f = h5py.File(filename, 'r')
    # default_goals = np.array(f['test_set'])
    # print(default_goals)
    # ------------ HYPERPARAMS --------------
    ENV = 'FetchReach-v1'
    # ENV = 'FetchPickAndPlace-v1'
    # ENV = 'FetchPush-v1'
    VERBOSE = False
    ACTOR_LEARNING_RATE = 1e-3
    CRITIC_LEARNING_RATE = 1e-3
    BUFFER_CAPACITY = int(1e6)
    BATCH_SIZE = 512
    NOISE_TYPE = "Normal"
    NOISE_STDDEV = 0.1
    TAU = 1e-3
    GAMMA = 0.98
    N_EPOCHS = 200
    N_CYCLES = 10
    N_EPISODES_PER_CYCLES = 16
    N_OPTIMIZATION_STEPS = 50
    N_REPLAYED_GOALS = 4
    NORMALIZED_OBSERVATIONS = True

    # create the environment
    env = gym.make(ENV)
    env.seed(SEED)


    obs_shape = env.observation_space.spaces['observation'].shape[0]
    action_shape = env.action_space.low.size
    goal_shape = env.observation_space.spaces['desired_goal'].low.size
    max_action_value = float(env.action_space.high[0])
    min_action_value = float(env.action_space.low[0])
    action_val = env.action_space.high


    success_rate = []


    print('Information about the environment:')
    print('...Observation dimension: ', obs_shape)
    print('...Goal dimension: ', goal_shape)
    print('...Action dimension: ', action_shape)
    print('...Max value for the action:', max_action_value)
    print('...Min value for the action:', min_action_value)
    print('...Distance threshold:', env.env.distance_threshold)


    # we have to initialize the off-policy RL algorithm
    agent = DDPG_Agent(env, obs_shape, goal_shape, action_shape, max_action_value, min_action_value,
                       replay_buffer_capacity=BUFFER_CAPACITY, actor_lr = ACTOR_LEARNING_RATE, critic_lr= CRITIC_LEARNING_RATE, batch_size=BATCH_SIZE, tau=TAU, gamma=GAMMA,
                       noise_stddev=NOISE_STDDEV, noise_type=NOISE_TYPE, normalized_observations=NORMALIZED_OBSERVATIONS, seed=SEED)


    # decide the sampling strategy
    SAMPLING_STRATEGY = "Future"
    assert SAMPLING_STRATEGY == "Final" or SAMPLING_STRATEGY == "Future", "Check the value of SAMPLING_STRATEGY, only Future or Final are accepted"

    n_success = 0
    n_episode = 0
    tot_episodes = N_EPOCHS * N_CYCLES * N_EPISODES_PER_CYCLES
    episode_per_epoch = N_CYCLES * N_EPISODES_PER_CYCLES
    print('\nStart training...')
    for _ep in range(N_EPOCHS):
        n_success_epoch = 0
        for _cycle in range(N_CYCLES):
            done = False
            trajectory = Trajectory()
            for _episode in range(N_EPISODES_PER_CYCLES):
                n_episode += 1
                # print("\r   Episode " + str(n_episode) + "/" + str(tot_episodes), end="          ")
                state = env.reset()
                agent.reset()
                sample_new_goal(env, state, X_RANGE, Y_RANGE, Z_RANGE)
                # obs = state['observatioszn']
                goal = state['desired_goal']
                # print(goal)
                # trajectory, where I store everything I need
                i = 0
                solved_once = False
                while not done:
                    i += 1
                    obs = state['observation']
                    # print(obs)
                    # we choose the action
                    # we have to find a good balance between exploration adn exploitation
                    if _ep < round(N_EPOCHS/3):
                        prob = np.random.rand()
                        if prob <= 0.7:
                            action = env.action_space.sample()
                        else:
                            action = agent.select_action(obs, goal, add_noise=True, use_target_network=False)
                    else:
                        prob = np.random.rand()
                        if prob <= 0.2:
                            action = env.action_space.sample()
                        else:
                            action = agent.select_action(obs, goal, add_noise=True, use_target_network=False)

                    # action = agent.select_action(obs, goal, add_noise=True)

                    # we execute the action
                    next_obs, reward, done, info = env.step(action)
                    if reward == 0.0:
                        solved_once = True
                        # we reach the target --> problem: we can reach the target more than once
                        # n_success +=1
                        # n_success_epoch += 1

                    achieved_goal = next_obs['achieved_goal']

                    # now we can add this transition to the trajectory
                    trajectory.add_transition(observations=obs,
                                              desired_goals=goal,
                                              actions=action,
                                              achieved_goals=achieved_goal,
                                              next_observations=next_obs['observation'],
                                              rewards=reward,
                                              done=done,
                                              infos=info)
                    if VERBOSE and done:
                        print(i)

                    state = next_obs
                    #
                    # do optimization step
                    # if agent.replay_buffer.size() > BATCH_SIZE:
                    #     agent.learn()

                if solved_once:
                    n_success_epoch +=1

                # at this point we have a trajectory of length 50
                assert len(trajectory['observations']) == 50, "Number of transitions inside the trajectory are more than 50"

                if HER_SAMPLING:
                    # for each transition in the trajectory we have to sample a new goal
                    if SAMPLING_STRATEGY == "Final":
                        sampled_transitions = final_sampling_strategy(env, trajectory)
                        assert len(sampled_transitions['observations']) == 50, "problem with the sampled transition (Final strategy)"
                    else:
                        sampled_transitions = future_sampling_strategies(env, trajectory, N_REPLAYED_GOALS)
                        assert len(sampled_transitions['observations']) == 50 * N_REPLAYED_GOALS, "problem with the sampled transition (Future stategy)"

                # now we have to store these trajectories in the replay buffer
                agent.replay_buffer.add_trajectory(trajectory, NORMALIZED_OBSERVATIONS, agent.obs_stats, agent.goal_stats)
                # agent.store_trajectory(trajectory)

                if HER_SAMPLING:
                    agent.replay_buffer.add_trajectory(sampled_transitions, NORMALIZED_OBSERVATIONS, agent.obs_stats, agent.goal_stats)
                    # agent.store_trajectory(sampled_transitions)

                # at this point we have stored both the original trajectory and the sampled transitions in the replay buffer
                # now, after 16 episodes we have to do 40 optimization steps
                # ppp = 0
                # if _episode in [15,30,49]:

            if agent.replay_buffer.size() > BATCH_SIZE and _cycle in [3,6,9]:
                # print('wwf')
                for _ in range(N_OPTIMIZATION_STEPS):
                    # print('fff')
                    agent.learn()
                # print(ppp)
        # print(n_success_epoch)
        # print(N_EPISODES_PER_CYCLES*N_CYCLES)
        # print('{} Epochs, {} Episodes, Success Rate: {}, Success Rate Epoch: {}'.format(_ep+1, n_episode, n_success / n_episode, n_success_epoch/(N_EPISODES_PER_CYCLES*N_CYCLES)))

        # at each epoch we run the policy withouth the noise to solve the test goal
        if ENV == 'FetchReach-v1':
            with torch.no_grad():
                success = 0
                # print(success)
                for i in range(N_TEST_EXAMPLES):
                    obs = env.reset()
                    env.env.goal = test_goals[i]
                    obs['desired_goal'] = env.env.goal
                    goal = obs['desired_goal']
                    done = False
                    solved = False

                    while not done:
                        state = obs['observation']
                        action = agent.select_action(state, goal, add_noise=False,  use_target_network=False)
                        next_obs, reward, done, info = env.step(action)
                        if reward == 0.0:
                            done = True
                            solved = True
                            success +=1
                        obs = next_obs
                        assert np.array_equal(next_obs['desired_goal'], goal), "Problem with the goal assignment"

                    if _ep > 35 and not solved and VERBOSE:
                        print(goal)

            with torch.no_grad():
                v_success = 0
                # print(v_success)
                for i in range(N_VALID_EXAMPLE):
                    obs = env.reset()
                    env.env.goal = validation_goal[i]
                    obs['desired_goal'] = env.env.goal
                    goal = obs['desired_goal']
                    done = False

                    while not done:
                        state = obs['observation']
                        action = agent.select_action(state, goal, add_noise=False,  use_target_network=False)
                        next_obs, reward, done, info = env.step(action)
                        if reward == 0.0:
                            done = True
                            v_success +=1
                        obs = next_obs
                        assert np.array_equal(next_obs['desired_goal'], goal), "Problem with the goal assignment"

            with torch.no_grad():
                # here we use the default sampling goal
                default_success = 0
                # print(n_success)
                for i in range(len(default_goals)):
                    obs = env.reset()
                    env.env.goal = default_goals[i]
                    obs['desired_goal'] = env.env.goal
                    goal = obs['desired_goal']
                    done = False

                    while not done:
                        state = obs['observation']
                        action = agent.select_action(state, goal, add_noise=False, use_target_network=False)
                        next_obs, reward, done, info = env.step(action)
                        if reward == 0.0:
                            done = True
                            default_success += 1
                        obs = next_obs
                        assert np.array_equal(next_obs['desired_goal'], goal), "Problem with the goal assignment"

            print('Epoch {}, Training Episodes: {}, Success Valid Rate Episodes: {}, Success Test Rate Epoch: {}, Success Rate Default Goal Sampling: {}'.format(_ep + 1, n_episode, v_success/N_VALID_EXAMPLE, success/N_TEST_EXAMPLES, default_success/len(default_goals)))
            success_rate.append(default_success / len(default_goals))

        # elif ENV == 'FetchPush-v1':
        #
        #     with torch.no_grad():
        #         # here we use the default sampling goal
        #         default_success = 0
        #         # print(n_success)
        #         for i in range(len(push_goal)):
        #             obs = env.reset()
        #             env.env.goal = push_goal[i]
        #             obs['desired_goal'] = env.env.goal
        #             goal = obs['desired_goal']
        #             done = False
        #
        #             while not done:
        #                 state = obs['observation']
        #                 action = agent.select_action(state, goal, add_noise=False, use_target_network=False)
        #                 next_obs, reward, done, info = env.step(action)
        #                 if reward == 0.0:
        #                     done = True
        #                     default_success +=1
        #                 obs = next_obs
        #                 assert np.array_equal(next_obs['desired_goal'], goal), "Problem with the goal assignment"
        #
        #     print('Epoch {}, Training Episodes: {},  Success Test Rate Epoch: {}'.format(_ep + 1, n_episode, default_success/len(push_goal)))
        #     success_rate.append(default_success/len(push_goal))
        #
        # else: #env=pick and place
        #
        #     with torch.no_grad():
        #         # here we use the default sampling goal
        #         default_success = 0
        #         # print(n_success)
        #         for i in range(len(pick_and_place_goals)):
        #             obs = env.reset()
        #             env.env.goal = pick_and_place_goals[i]
        #             obs['desired_goal'] = env.env.goal
        #             goal = obs['desired_goal']
        #             done = False
        #
        #             while not done:
        #                 state = obs['observation']
        #                 action = agent.select_action(state, goal, add_noise=False, use_target_network=True)
        #                 next_obs, reward, done, info = env.step(action)
        #                 if reward == 0.0:
        #                     done = True
        #                     default_success += 1
        #                 obs = next_obs
        #                 assert np.array_equal(next_obs['desired_goal'], goal), "Problem with the goal assignment"
        #
        #     print('Epoch {}, Training Episodes: {},  Success Test Rate Epoch: {}'.format(_ep + 1, n_episode,
        #                                                                                  default_success / len(pick_and_place_goals)))


    torch.save(agent.actor.state_dict(), '../HER/final_checkpoint/checkpoint_actor_reach_bottom_seed_{}.pth'.format(SEED))
    torch.save(agent.actor_target.state_dict(), '../HER/final_checkpoint/checkpoint_actor_target_reach_bottom_seed_{}.pth'.format(SEED))
    torch.save(agent.critic.state_dict(), '../HER/final_checkpoint/checkpoint_critic_reach_bottom_seed_{}.pth'.format(SEED))
    torch.save(agent.critic_target.state_dict(), '../HER/final_checkpoint/checkpoint_critic_target_reach_bottom_seed_{}.pth'.format(SEED))

    print(success_rate)
    # store the performances
    file_path_test = "../HER/final_checkpoint/reach_seed_bootom_bottom_more_exploration_02_all_epochs_final_strategy_seed_{}".format(SEED)

    file_5 = h5py.File(file_path_test, 'w')

    file_5.create_dataset('success_rate', data=success_rate, chunks = True, compression = "gzip")

    file_5.close()

    env.close()



# at this point we have that Fetch is trained, so we can check if it really works
# obs = env.reset()
# goal = obs['desired_goal']
# done_ = False
# reward = -100
# while reward != 0:
#     env.render()
#     state = obs['observation']
#     action = agent.select_action(state, goal, add_noise=False)
#     next_obs, reward, done, info = env.step(action)
#     print(reward)
#     obs = next_obs















