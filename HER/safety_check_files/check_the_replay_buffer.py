import numpy as np
from HER.replay_buffer import ReplayBuffer

buffer = ReplayBuffer(1,1,1,5)

traj = dict(
    observations = np.array([1,2,3,4,5]).reshape(-1,1),
    actions = np.array([1,2,3,4,5]).reshape(-1,1),
    rewards = np.array([1,2,3,4,5]),
    next_observations = np.array([1,2,3,4,5]).reshape(-1,1),
    desired_goals = np.array([1,2,3,4,5]).reshape(-1,1),
    achieved_goals = np.array([1,2,3,4,5]).reshape(-1,1),
    info = np.array([1,2,3,4,5]))

buffer.add_trajectory(traj, False, None, None)
print('safety chaeck')
print(buffer._observations)

traj = dict(
    observations = np.array([6]).reshape(-1,1),
    actions = np.array([6]).reshape(-1,1),
    rewards = np.array([6]),
    next_observations = np.array([6]).reshape(-1,1),
    desired_goals = np.array([6]).reshape(-1,1),
    achieved_goals = np.array([6]).reshape(-1,1),
    info = np.array([6]))

buffer.add_trajectory(traj, False, None, None)
print('safety check')
print(buffer._observations)


traj = dict(
    observations = np.array([7,8,9]).reshape(-1,1),
    actions = np.array([7,8,9]).reshape(-1,1),
    rewards = np.array([7,8,9]),
    next_observations = np.array([7,8,9]).reshape(-1,1),
    desired_goals = np.array([7,8,9]).reshape(-1,1),
    achieved_goals = np.array([7,8,9]).reshape(-1,1),
    info = np.array([7,8,9]))

buffer.add_trajectory(traj, False, None, None)
print('safety check')
print(buffer._observations)

traj = dict(
    observations = np.array([10,11]).reshape(-1,1),
    actions = np.array([10,11]).reshape(-1,1),
    rewards = np.array([10,11]),
    next_observations = np.array([10,11]).reshape(-1,1),
    desired_goals = np.array([10,11]).reshape(-1,1),
    achieved_goals = np.array([10,11]).reshape(-1,1),
    info = np.array([10,11]))

buffer.add_trajectory(traj, False, None, None)
print('safety check')
print(buffer._observations)
print(buffer._actions)
print(buffer._rewards)
print(buffer._next_observations)
print(buffer._desired_goals)


traj = dict(
    observations = np.array([12,13,14,15]).reshape(-1,1),
    actions = np.array([12,13,14,15]).reshape(-1,1),
    rewards = np.array([12,13,14,15]),
    next_observations = np.array([12,13,14,15]).reshape(-1,1),
    desired_goals = np.array([12,13,14,15]).reshape(-1,1),
    achieved_goals = np.array([12,13,14,15]).reshape(-1,1),
    info = np.array([12,13,14,15]))

buffer.add_trajectory(traj, False, None, None)
print('safety check')
print(buffer._observations)
print(buffer._actions)
print(buffer._rewards)
print(buffer._next_observations)
print(buffer._desired_goals)

traj = dict(
    observations = np.array([12,13,14,15]).reshape(-1,1),
    actions = np.array([12,13,14,15]).reshape(-1,1),
    rewards = np.array([12,13,14,15]),
    next_observations = np.array([12,13,14,15]).reshape(-1,1),
    desired_goals = np.array([12,13,14,15]).reshape(-1,1),
    achieved_goals = np.array([12,13,14,15]).reshape(-1,1),
    info = np.array([12,13,14,15]))

buffer.add_trajectory(traj, False, None, None)
print('safety check')
print(buffer._observations)
print(buffer._actions)
print(buffer._rewards)
print(buffer._next_observations)
print(buffer._desired_goals)