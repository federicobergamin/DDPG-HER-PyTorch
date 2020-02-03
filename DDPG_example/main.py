import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import DDPG_example.utils as utils
from DDPG_example.DDPG_Agent import DDPG_Agent
from copy import  copy

ENV = 'Pendulum-v0'
VERBOSE = False
env = gym.make(ENV).env
# env = gym.make('BipedalWalker-v2').env

# env.seed(10)
# TODO: it would be cool try to use a decreasing noise --> AND PARAMETERS ADAPTIVE NOISE
obs_shape = env.observation_space.shape[0]
action_shape = env.action_space.shape[0]
max_action_value = float(env.action_space.high[0])
min_action_value = float(env.action_space.low[0])

print('Info about the environment:')
print('Observation shape:', obs_shape)
print('Action shape:', action_shape)
print('Max value for the action:', max_action_value)
print('Min value for the action:', min_action_value)


scores_deque = deque(maxlen=100)
scores = []
max_scores = -np.Inf

N_RANDOM_EXP_EPISODE = 800
MAX_STEP_EXPL = 200
N_EPISODE = 200 # number of episodes
MAX_T = 200 # number of steps/transition in each episode
BATCH_SIZE = 128
BUFFER_CAPACITY = int(1e6)
# MIN_MEMORY_SIZE = 8000 # I start doing the learning after 2000 transition in the Replay_Buffer
NOISE_TYPE = 'Normal'


agent = DDPG_Agent(env, obs_shape, action_shape, max_action_value, min_action_value, replay_buffer_capacity=BUFFER_CAPACITY, batch_size=BATCH_SIZE, noise_type=NOISE_TYPE, normalized_observations=True, seed=2)
print(agent.replay_buffer.size())
print('Start collecting data for the exploration doing random actions')
iter = 0
for rndm_exp_episode in range(1, N_RANDOM_EXP_EPISODE+1):
    print("\r   Episode " + str(rndm_exp_episode) + "/" + str(N_RANDOM_EXP_EPISODE), end="          ")
    state = env.reset()

    for t in range(MAX_STEP_EXPL):
        # instead of following a policy, at the beginning we sample random actions
        action = env.action_space.sample()
        # print(action)
        # print(action.item())
        next_state, reward, done, _ = env.step(action)
        # print(done)
        agent.store_transitions(state, action, next_state, reward, done)

        # I should say that now state==next_state right=?=
        state = copy(next_state)
        iter += 1



print('safety check')
print(agent.replay_buffer.size())

print(N_RANDOM_EXP_EPISODE*MAX_STEP_EXPL)
print(N_RANDOM_EXP_EPISODE*(MAX_STEP_EXPL+1))
print(iter)
if N_RANDOM_EXP_EPISODE*MAX_STEP_EXPL <= BUFFER_CAPACITY:
    assert ((N_RANDOM_EXP_EPISODE)*(MAX_STEP_EXPL)) == agent.replay_buffer.size()

print('Start training...')
for i_episode in range(1, N_EPISODE + 1):
    print("\r   Episode " + str(i_episode) + "/" + str(N_EPISODE), end="          ")
    state = env.reset()
    agent.reset()
    episode_reward = 0
    for t in range(MAX_T):
        action = agent.select_action(state)
        # print(action)
        # env.render()
        # print(action)
        # print(action.item())
        next_state, reward, done, _ = env.step(action)
        # print(done)
        agent.store_transitions(state, action, next_state, reward, done)

        # If enough samples are available in memory we train the networks
        if agent.replay_buffer.size() > BATCH_SIZE:
            agent.learn()
        # old_state = state
        state = next_state
        episode_reward += reward

        if done or t==MAX_T:
            if VERBOSE:
                print('Episode: {}, Steps: {}, Reward: {}'.format(i_episode, t, episode_reward))
            break

    scores_deque.append(episode_reward)
    scores.append(episode_reward)
    if i_episode % 100 == 0:
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        # torch.save(agent.actor.state_dict(), '../DDPG_example/checkpoints/checkpoint_actor_{}_env_{}_episode_{}_noise_v2.pth'.format(ENV, i_episode, NOISE_TYPE))
        # torch.save(agent.critic.state_dict(), '../DDPG_example/checkpoints/checkpoint_critic_{}_env_{}_episode_{}_noise_v2.pth'.format(ENV, i_episode, NOISE_TYPE))

# torch.save(agent.actor.state_dict(), '../DDPG_example/checkpoints/checkpoint_actor_final_{}_env_{}_noise.pth'.format(ENV, NOISE_TYPE))
# torch.save(agent.critic.state_dict(), '../DDPG_example/checkpoints/checkpoint_critic_final_{}_env_{}_noise.pth'.format(ENV, NOISE_TYPE))

# ----- see the learned policy in actions
state = env.reset()
agent.reset()
episode_reward = 0
for t in range(400):
    action = agent.select_action(state, add_noise=False)
    env.render()
    next_state, reward, done, _ = env.step(action)
    state = next_state
    episode_reward += reward

    if done:
        print('Episode reward:', episode_reward)
        break

env.close()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()







