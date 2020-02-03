import _pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import h5py
from HER.plotting import update_one_plot, make_figure

# we have to load all the performance
path = "../HER/performance_up/reach_seed_up_UP_1"
file = h5py.File(path, 'r')
seed_1 = np.array(file['success_rate'])

path = "../HER/performance_up/reach_seed_up_UP_42"
file = h5py.File(path, 'r')
seed_2 = np.array(file['success_rate'])

path = "../HER/performance_up/reach_seed_up_UP_213"
file = h5py.File(path, 'r')
seed_3 = np.array(file['success_rate'])

path = "../HER/performance_up/reach_seed_up_UP_666"
file = h5py.File(path, 'r')
seed_4 = np.array(file['success_rate'])

path = "../HER/performance_up/reach_seed_up_UP_777"
file = h5py.File(path, 'r')
seed_5 = np.array(file['success_rate'])
# print(seed_5)

success_rate_02_future = []
success_rate_02_future.append(seed_1)
success_rate_02_future.append(seed_2)
success_rate_02_future.append(seed_3)
success_rate_02_future.append(seed_4)
success_rate_02_future.append(seed_5)
success_rate_02_future = np.array(success_rate_02_future)


# we have to load all the performance
path = "../HER/performance_up/reach_seed_up_UP_more_exploration_07_for_third_of_epochs_1"
file = h5py.File(path, 'r')
seed_1 = np.array(file['success_rate'])

path = "../HER/performance_up/reach_seed_up_UP_more_exploration_07_for_third_of_epochs_42"
file = h5py.File(path, 'r')
seed_2 = np.array(file['success_rate'])

path = "../HER/performance_up/reach_seed_up_UP_more_exploration_07_for_third_of_epochs_213"
file = h5py.File(path, 'r')
seed_3 = np.array(file['success_rate'])

path = "../HER/performance_up/reach_seed_up_UP_more_exploration_07_for_third_of_epochs_666"
file = h5py.File(path, 'r')
seed_4 = np.array(file['success_rate'])

path = "../HER/performance_up/reach_seed_up_UP_more_exploration_07_for_third_of_epochs_777"
file = h5py.File(path, 'r')
seed_5 = np.array(file['success_rate'])
# print(seed_5)

success_rate_07_Future = []
success_rate_07_Future.append(seed_1)
success_rate_07_Future.append(seed_2)
success_rate_07_Future.append(seed_3)
success_rate_07_Future.append(seed_4)
success_rate_07_Future.append(seed_5)
success_rate_07_Future = np.array(success_rate_07_Future)


# we have to load all the performance
path = "../HER/performance_up/reach_seed_up_UP_more_exploration_02_all_epochs_final_strategy_seed_1"
file = h5py.File(path, 'r')
seed_1 = np.array(file['success_rate'])

path = "../HER/performance_up/reach_seed_up_UP_more_exploration_02_all_epochs_final_strategy_seed_42"
file = h5py.File(path, 'r')
seed_2 = np.array(file['success_rate'])

path = "../HER/performance_up/reach_seed_up_UP_more_exploration_02_all_epochs_final_strategy_seed_213"
file = h5py.File(path, 'r')
seed_3 = np.array(file['success_rate'])

path = "../HER/performance_up/reach_seed_up_UP_more_exploration_02_all_epochs_final_strategy_seed_666"
file = h5py.File(path, 'r')
seed_4 = np.array(file['success_rate'])

path = "../HER/performance_up/reach_seed_up_UP_more_exploration_02_all_epochs_final_strategy_seed_777"
file = h5py.File(path, 'r')
seed_5 = np.array(file['success_rate'])
# print(seed_5)

success_rate_02_final = []
success_rate_02_final.append(seed_1)
success_rate_02_final.append(seed_2)
success_rate_02_final.append(seed_3)
success_rate_02_final.append(seed_4)
success_rate_02_final.append(seed_5)
success_rate_02_final = np.array(success_rate_02_final)


# we have to load all the performance
path = "../HER/performance_up/reach_seed_up_UP_more_exploration_07_for_third_of_epochs_final_strategy_seed_1"
file = h5py.File(path, 'r')
seed_1 = np.array(file['success_rate'])

path = "../HER/performance_up/reach_seed_up_UP_more_exploration_07_for_third_of_epochs_final_strategy_seed_42"
file = h5py.File(path, 'r')
seed_2 = np.array(file['success_rate'])

path = "../HER/performance_up/reach_seed_up_UP_more_exploration_07_for_third_of_epochs_final_strategy_seed_213"
file = h5py.File(path, 'r')
seed_3 = np.array(file['success_rate'])

path = "../HER/performance_up/reach_seed_up_UP_more_exploration_07_for_third_of_epochs_final_strategy_seed_666"
file = h5py.File(path, 'r')
seed_4 = np.array(file['success_rate'])

path = "../HER/performance_up/reach_seed_up_UP_more_exploration_07_for_third_of_epochs_final_strategy_seed_777"
file = h5py.File(path, 'r')
seed_5 = np.array(file['success_rate'])
# print(seed_5)

success_rate_07_final = []
success_rate_07_final.append(seed_1)
success_rate_07_final.append(seed_2)
success_rate_07_final.append(seed_3)
success_rate_07_final.append(seed_4)
success_rate_07_final.append(seed_5)
success_rate_07_final = np.array(success_rate_07_final)


index = [i for i in range(len(seed_1))]

# print(index)
f, ax = plt.subplots()
# ax = plt.gca()
# update_one_plot(ax, "#0072BD", 'epsilon = 0.2, strategy = Future', index, success_rate_02_future, 'median')
# update_one_plot(ax, "#D95319", 'epsilon = 0.7 for 1/3 epochs, strategy = Future', index, success_rate_07_Future, 'median')
update_one_plot(ax, "#7E2F8E", 'epsilon = 0.2, strategy = Final', index, success_rate_02_final, 'median')
update_one_plot(ax, "#006450", 'epsilon = 0.7 for 1/3 epochs, strategy = Final', index, success_rate_07_final, 'median')
# update_one_plot(ax, "#7E2F8E", 'Path heuristic', index, np.transpose(explored_path), 'median')
# update_one_plot(ax, "#EDB120", 'Path heuristic more accurate', index, np.transpose(explored_path_mc), 'median')
plt.xticks(np.arange(0,index[-1]+2,step=50))
plt.ylabel('Success Rate')
plt.xlabel('Epochs')
plt.title('Success rate of the policy trained in the task space upper half ')
plt.legend(loc='lower right')
plt.grid()
plt.show()
f.savefig("upper_policy_07vs02_final.pdf")
# f.savefig('FSFWFFWFWFFW.pdf')

