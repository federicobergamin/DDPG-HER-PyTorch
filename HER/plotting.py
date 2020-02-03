import os

import matplotlib.pyplot as plt
import numpy as np

COLORS = ["#0072BD",
          "#D95319",
          "#006450",
          "#7E2F8E",
          "#77AC30",
          "#EDB120",
          "#4DBEEE",
          "#A2142F",
          "#191970",
          "#A0522D"]
ALPHA = "0.2"


def make_figure(figsize=(10, 6)):
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    return fig, ax

def make_subplots(dims, figsize=(10,6)):
    assert len(dims)==2
    fig, axarr = plt.subplots(dims[0], dims[1], figsize=figsize)
    axarr = axarr.flat

    # for ax in axarr:
        # ax.spines["top"].set_visible(False)
        # ax.spines["right"].set_visible(False)
        # ax.get_xaxis().tick_bottom()
        # ax.get_yaxis().tick_left()
    return fig, axarr


def update_one_plot(ax, color, label, x_data, y_data, central_tend):
    """

    :param ax: axes
    :param color: one color
    :param label: label for this series
    :param x_data: list or array with x data
    :param y_data: list of lists - each list contains a set of y data for this series, to be averaged
    :param central_tend: central tendency measure - 'mean' or 'median'
    :return:
    """
    x = list(x_data)
    y_mean = []
    y_upper = []
    y_lower = []
    max_len = max([len(x_) for x_ in y_data])
    for i in range(max_len):
        y_data = [item for item in y_data if i<len(item)]
        values = [lst[i] for lst in y_data]
        if central_tend== 'mean':
            y_mean.append(np.mean(values))
            y_lower.append(y_mean[-1] - np.std(values))
            y_upper.append(y_mean[-1] + np.std(values))
        elif central_tend== 'median':
            y_mean.append(np.median(values))
            y_lower.append(np.percentile(values, 25))
            y_upper.append(np.percentile(values, 75))
    ax.fill_between(np.array(x), np.array(y_lower), np.array(y_upper), color=color, alpha=ALPHA)
    ax.plot(x, y_mean, color=color, label=label, lw=1)


def get_file_groups(filenames):
    """Create dict with key=label, value=listOfFilenames, and return all values (list of lists of filenames)"""
    groups = dict()
    for fname in filenames:
        label = get_label(fname)
        if label not in groups:
            groups[label] = []
        groups[label].append(fname)
    return groups.values()

def get_label(filename):
    return filename.split("_")[-1]

def plottable_files_in(folder):
    return [f for f in os.listdir(folder) if is_plottable_file(folder, f)]

def is_plottable_file(folder, filename):
    return (os.path.isfile(os.path.join(folder, filename))) and \
           not (filename.startswith('.') or filename.startswith('_'))


def sort_fname_groups(groups_unsorted, label_converter={}):
    #input is list of lists of strings
    sorter = []
    for group in groups_unsorted:
        label = get_label(group[0])
        if label in label_converter:
            label = label_converter[label]
        sorter.append((label, group))
    sorter.sort()
    print(sorter)
    return [group for (label, group) in sorter]