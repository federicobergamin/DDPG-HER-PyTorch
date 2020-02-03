import numpy as np
import torch

# np.random.seed(42)
# torch.manual_seed(42)

def hidden_layer_init(hidden_layer):
    ''' We follow the initialization of the original paper:
        both weights and bias are initialized using a uniform distribution
        [-1/np.sqrt(f),1/np.sqrt(f)] where f is the fan_in of the layer '''
    # we start computing the fan in
    fan_in = hidden_layer.weight.data.size()[0]
    # fan_in = torch.nn.init._calculate_correct_fan(hidden_layer.weight, mode='fan_in')
    # torch.nn.init.uniform_(hidden_layer.weight, - 1. / np.sqrt(fan_in), 1. / np.sqrt(fan_in))
    # torch.nn.init.uniform_(hidden_layer.bias, -1. / np.sqrt(fan_in), 1. / np.sqrt(fan_in))
    lim = 1. / np.sqrt(fan_in)
    hidden_layer.weight.data.uniform_(-lim, lim)
    # hidden_layer.bias.data.uniform_(-lim, 1. /lim)


def output_layer_init(output_layer):
    ''' We use the same initialization as in the original paper'''
    # torch.nn.init.uniform_(output_layer.weight, -3 * 10e-4, 3 * 10e-4)
    # torch.nn.init.uniform_(output_layer.bias, -3 * 10e-4, 3 * 10e-4)
    lim= 3e-3
    output_layer.weight.data.uniform_(-lim, lim)
    output_layer.bias.data.uniform_(-lim, lim)


def sample_new_goal(environment, instance, x_range, y_range, z_range):
    ''' Each range is an array with two elements [min, max]'''
    assert len(x_range) == 2 and len(y_range) == 2 and len(z_range) == 2, 'A range you passed has not exactly 2 numbers'
    goal =  np.array([np.random.uniform(x_range[0],x_range[1]), np.random.uniform(y_range[0],y_range[1]), np.random.uniform(z_range[0],z_range[1])])
    environment.env.goal = goal
    instance['desired_goal'] = environment.env.goal
