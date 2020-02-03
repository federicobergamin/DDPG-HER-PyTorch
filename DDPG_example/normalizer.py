import numpy as np
import torch

# from DDPG_example.DDPG import normalize

####
# we need methods to compute the normalization and keep tracking of the
class Running_stats(object):
    ''' This class normalize the data and also keep a moving average of the
        mean of each samples and of the standard deviation.
        Combination of the code of Marcin Andrychowicz and Vitchyr Pong '''

    def __init__(self, colnums, eps = 1e-2, mean = 0, stddev = 1, default_clip_range = np.inf):
        assert stddev > 0
        stddev = stddev + eps
        self.colnums = colnums
        self.eps = eps
        self.sum = np.zeros(self.colnums, np.float32)
        self.squared_sum = np.zeros(self.colnums, np.float32)
        self.count = np.ones(1, np.float32)
        self.running_mean = mean + np.zeros(self.colnums, np.float32)
        self.running_stddev = stddev * np.ones(self.colnums, np.float32)
        self.updated = True
        self.clip_range = default_clip_range



    def update_stats(self, new_observation):
        # shape = new_observation.shape[0]
        self.sum += new_observation
        self.squared_sum += new_observation**2
        self.running_mean = self.sum / self.count
        # print(((self.squared_sum / self.count)-self.running_mean**2))
        self.running_stddev = np.sqrt(np.clip(((self.squared_sum / self.count)-self.running_mean**2), a_min=self.eps, a_max=None))
        self.count += 1

def normalize(observation, statistics):
    if statistics is None:
        return observation
    else:
        return (observation - statistics.running_mean) / statistics.running_stddev

def normalizer_example():
    matrix = np.array([[200,10,1000,50,3],[250,11,1029,45,5],[300,18,1045,42,3],[400,21,1100,37,8]])
    running_stats = Running_stats(matrix.shape[1])

    for obs in matrix:
        running_stats.update_stats(obs)
        print(running_stats.running_mean, running_stats.running_stddev)

    obs = np.array([225,19,1050,38,2])
    obs_norm = normalize(obs, running_stats)

    print(obs_norm)


# def normalizer_example():
#     array =
if __name__=='__main__':
    normalizer_example()


