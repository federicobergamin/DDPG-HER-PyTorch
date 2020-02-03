''' OpenAI: adding adaptive noise to the parameters of reinforcement learning algorithms
    frequently boosts performance.
    In addition to that: Traditional RL uses action space noise to change the likelihoods
    associated with each action the agent might take from one moment to the next.
    Parameter space noise injects randomness directly into the parameters of the agent,
    altering the types of decisions it makes such that they always
    fully depend on what the agent currently senses.

    Most of the code is taken by OpenAI Baselines
'''

# we are going to create both action-space noise and parameter-space noise

import numpy as np
from copy import copy
import random

# np.random.seed(42)


class Adaptive_Params_Noise(object):

    def __init__(self, initial_stddev=0.1, desired_action_stddev=0.1, adaption_coefficient=1.01):
        self.initial_stddev = initial_stddev
        self.desired_action_stddev = desired_action_stddev
        self.adaption_coefficient = adaption_coefficient

        # set the current stddev to the initial one
        self.current_stddev = initial_stddev

    def adapt(self, distance):
        # TODO: CHECK HOW TO COMPUTE THIS DISTANCE
        if distance > self.desired_action_stddev:
            # we should decrease the standard deviation
            self.current_stddev /= self.adaption_coefficient
        else:
            # we should increase the standard deviation
            self.current_stddev *= self.adaption_coefficient

    def get_current_stddev(self):
        stats = {
            'param_noise_stddev': self.current_stddev
        }
        return stats

    def __repr__(self):
        return 'Adaprive_Param_Noise(initial_stddev={}, desired_action_stddev={}, adoption_coefficient={})'.format(
            self.initial_stddev, self.desired_action_stddev, self.adaption_coefficient)


class Action_Noise(object):

    def reset(self):
        pass


class Normal_Action_Noise(Action_Noise):

    def __init__(self, mean, variance):
        self.mean = mean
        self.variance = variance

    def sample(self):
        return np.random.normal(self.mean, self.variance)

    def __repr__(self):
        return 'Normal_Action_Noise(mu={}, sigma={})'.format(self.mean, self.variance)


###  this was not so precise
# class Ornstein_Uhlenbeck_Action_Noise(Action_Noise):
#
#     def __init__(self, mean, variance, theta = 0.15, dt=1e-2, x0 = None):
#         self.mean = mean
#         self.variance = variance
#         self.theta = theta
#         self.dt = dt
#         self.x0 = x0
#         self.reset()
#
#     def __call__(self):
#         x = self.x_prev + self.theta * (self.mean - self.x_prev) * self.dt + self.variance * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
#         self.x_prev = x
#         return x
#
#     def reset(self):
#         self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mean)
#
#     def __repr__(self):
#         return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mean, self.variance)


class Ornstein_Uhlenbeck_Action_Noise(Action_Noise):
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mean=0., theta=0.15, variance=0.2):
        """Initialize parameters and noise process."""
        self.mean = mean * np.ones(size)
        self.theta = theta
        self.variance = variance
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy(self.mean)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mean - x) + self.variance * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state





























