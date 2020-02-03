''' In this file we implement a Trajectory class.
    This would be a dictionary that store all the observation, actions, next_observation, and done.
    In this way, when we sample a new goal, we have all the information here.

    This idea is taken from https://github.com/vitchyr/rlkit
'''

import numpy as np
# TODO: BEFORE USING IT, I SHOULD TRY THAT EVERYTHING WORKS PROPERLY

class Trajectory(dict):

    # constructor
    def __init__(self):
        super().__init__()
        self.trajectory_length = 0


    def add_transition(self, **transition):
        ''' Note that the transition implies something like
            observations = ...
            actions = ...
            next_observations = ...
            rewards = ...
            dones = ...
        '''
        for key, value in transition.items():
            # print(key)
            # print(value)
            # we should create the lists
            if key not in self:
                self[key] = [value]
                # print(self[key])
            else:
                self[key].append(value)
                # print(self[key])

        self.trajectory_length += 1

    def get_complete_trajectory(self):
        ''' It returns the dictionary '''

        final_trajectory = dict()

        for key, value in self.items():
            final_trajectory[key] = get_array(value)

        return final_trajectory


def get_array(list):
    ''' we need the if in case we have a dict of sub-dict'''
    if isinstance(list[0], dict):
        return list
    else:
        return np.array(list)


def trajectory_example():
    trajectory = Trajectory()
    trajectory.add_transition(observations = 1,
                              actions = 1,
                              next_observations = 1,
                              rewards = 1,
                              done = False)

    # print(trajectory.get_final_transition())

    trajectory.add_transition(observations = 2,
                              actions = 2,
                              next_observations = 2,
                              rewards = 2,
                              done=False)

    trajectory.add_transition(observations = 3,
                              actions = 3,
                              next_observations = 3,
                              rewards = 3,
                              done=True)

    fin = trajectory.get_complete_trajectory()

    print(fin)

if __name__=='__main__':
    trajectory_example()








