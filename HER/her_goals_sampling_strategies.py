''' In this file we try to implement two methods for sampling new goals given a trajectory.
    We will implement two out of four strategies described in https://arxiv.org/pdf/1707.01495.pdf:
    - final: for each transition the resamopled goal is the one that is satisfied in the final state of the trajectory
    - future: for each transition we sample K goals in the range next_obs - final_obs
'''

import numpy as np
from HER.trajectory import Trajectory

# we want to sample
def final_sampling_strategy(env, trajectory):
    # we get the initial trajectory as parameter
    observations = trajectory['observations']
    actions = trajectory['actions']
    rewards = trajectory['rewards']
    next_observations = trajectory['next_observations']
    desired_goals = trajectory['desired_goals']
    achieved_goals = trajectory['achieved_goals']
    info = trajectory['infos']
    done = trajectory['done']

    sampled_transitions = Trajectory()
    # for each transition in the trajectory, we have to sample the final episode achieved goal
    # and use that one as desired goal for all transitions --> we also have to recompute the goal
    # final goal achieved in the final state
    final_goal_episode = achieved_goals[-1]
    # print(final_goal_episode)
    for i in range(len(observations)):
        # we have to recompute the reward before being able to add the transition
        new_reward = env.compute_reward(achieved_goals[i], final_goal_episode, info[i])
        sampled_transitions.add_transition(observations=observations[i],
                                           actions=actions[i],
                                           desired_goals=final_goal_episode,
                                           rewards=new_reward,
                                           next_observations=next_observations[i],
                                           info=info[i],
                                           done=done[i])

    return sampled_transitions




def future_sampling_strategies(env, trajectory, n_sampled_goal):
    observations = trajectory['observations']
    actions = trajectory['actions']
    rewards = trajectory['rewards']
    next_observations = trajectory['next_observations']
    desired_goals = trajectory['desired_goals']
    achieved_goals = trajectory['achieved_goals']
    info = trajectory['infos']
    done = trajectory['done']

    sampled_transitions = Trajectory()
    T = len(observations)

    # in this case we follow the future strategy
    for i in range(T):
        # if len(observations) - i >= n_sampled_goal:
            # we are in the case we can sample 4 goals without having copies
        for _ in range(n_sampled_goal):
        #     we have to sample a goal from i to the end
            # i guess that when we reach end-i<n_sampled_goal,
            # we sample that one? or what? we sample n_sampled_goal time the same thing
            if i == len(observations): # last one
                sampled_goal_idx = i
            # elif len(observations) - i <  n_sampled_goal:
            #     sampled_goal_idx =
            else:
            #     sampled_goal_idx = np.random.randint(i, len(observations))
                sampled_goal_idx = np.random.randint(i, len(observations))
            # if i == T-1:
            #     print(sampled_goal_idx)

            # now we have to recompute the reward
            new_reward = env.compute_reward(achieved_goals[i], achieved_goals[sampled_goal_idx], info[i])

            sampled_transitions.add_transition(observations=observations[i],
                                               actions=actions[i],
                                               desired_goals=achieved_goals[sampled_goal_idx],
                                               rewards=new_reward,
                                               next_observations=next_observations[i],
                                               info=info[i],
                                               achieved_goals=achieved_goals[i],
                                               done = done[i])
        # else: # we are in a case in which len(observations) - i < n_sampled_goal, so there can be copies, which I guess we want to avoid to have
        #     # print('%%%%%%%%%%%%%%%%')
        #     # print(len(observations) - i)
        #     # print('---')
        #     for idx in range(len(observations) - i):
        #         sampled_goal_idx = len(observations)-idx-1
        #         # print(sampled_goal_idx)
        #         new_reward = env.compute_reward(achieved_goals[i], achieved_goals[sampled_goal_idx], info[i])
        #
        #         sampled_transitions.add_transition(observations=observations[i],
        #                                            actions=actions[i],
        #                                            desired_goals=achieved_goals[sampled_goal_idx],
        #                                            rewards=new_reward,
        #                                            next_observations=next_observations[i],
        #                                            info=info[i],
        #                                            achieved_goals=achieved_goals[i],
        #                                            done=done[i])
            # print('%%%%%%%%%%%%%%%%')
    return sampled_transitions





