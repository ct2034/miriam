#!/usr/bin/env python2
import os
import random
import sys
from enum import Enum, unique
from itertools import product
from matplotlib import pyplot as plt
import numpy as np
import pickle
from adamsmap_eval.filename_verification import (
    is_eval_cen_file)


@unique
class Planner(Enum):
    RCBS = 0
    ECBS = 1
    ILP = 2


@unique
class Graph(Enum):
    ODRM = 0
    UDRM = 1
    GRID = 2


def make_nice_title(s):
    """
    Turn strings like `an_awful_one` in `An Awful One`.
    :type s: str
    """

    nice_s = s.replace("_", " ")
    nice_s = nice_s.capitalize()
    return nice_s


def combinations_sort_key(item):
    g = Graph[item.split("-")[1]].value
    p = Planner[item.split("-")[0]].value
    return g * 3 + p


def how_may_datas(eval_results):
    dims = [0] * 3
    dims[0] = len(eval_results.keys())
    allkeys = [[]] * 3
    allkeys[0] = eval_results.keys()
    for type in eval_results.keys():
        if dims[1] < len(eval_results[type].keys()):
            dims[1] = len(eval_results[type].keys())
            allkeys[1] = eval_results[type].keys()
        for combination in eval_results[type].keys():
            if dims[2] < len(eval_results[type][combination].keys()):
                dims[2] = len(eval_results[type][combination].keys())
                allkeys[2] = eval_results[type][combination].keys()
    lengths = np.zeros(dims)
    for t, c, a in product(*map(range, dims)):
        type = allkeys[0][t]
        combination = allkeys[1][c]
        agents_n = allkeys[2][a]
        try:
            lengths[t, c, a] = len(eval_results[type][combination][agents_n])
        except Exception:
            pass
    return lengths



if __name__ == "__main__":
    plt.style.use('bmh')
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    markers = ['s', 'X', 'o']
    linestyles = ['solid', 'dashed', 'dotted']

    folder = sys.argv[1]  # type: str
    assert folder.endswith("/"), "Please call with folder"

    fname_base = "z_200_4096.eval_cen"

    eval_results = None
    for fname in sorted(os.listdir(folder)):
        if fname.endswith(".pkl") and fname.startswith(fname_base):
            with open(folder + fname) as fb:
                if eval_results is None:
                    eval_results = pickle.load(fb)
                else:
                    additional_results = pickle.load(fb)
                    # type, combination, agents_n
                    for type in additional_results.keys():
                        if type not in eval_results.keys():
                            eval_results[type] = additional_results[type]
                        else:
                            for combination in additional_results[type].keys():
                                if combination not in eval_results[type].keys():
                                    eval_results[type][combination] = additional_results[type][combination]
                                else:
                                    for agents_n in additional_results[type][combination].keys():
                                        if agents_n not in eval_results[type][combination].keys():
                                            eval_results[type][combination][agents_n] = additional_results[type][combination][agents_n]
                                        else:
                                            eval_results[type][combination][agents_n] += additional_results[type][combination][agents_n]
            print("read ... {}".format(fname))
            lengths = how_may_datas(eval_results)
            print("sum: {}, min_len: {}, max_len: {}".format(lengths.sum(), lengths.min(), lengths.max()))

    assert eval_results is not None

    successful_fig_data = eval_results['successful']
    for fig_title in eval_results.keys():
        fig, ax = plt.subplots()
        fig_data = eval_results[fig_title]
        lines = []
        combination_names = sorted(fig_data.keys(), key=combinations_sort_key)
        for combination_name in combination_names:
            i_g = Graph[combination_name.split("-")[1]].value
            i_p = Planner[combination_name.split("-")[0]].value
            combination_data = fig_data[combination_name]
            agents_ns_strs = combination_data.keys()
            agents_ns_strs = sorted(agents_ns_strs, key=int)[:-1]
            x = []
            if fig_title == 'successful':
                to_plot = map(
                    lambda k: 100. * np.count_nonzero(np.array(combination_data[k]))
                              / len(combination_data[k]),
                    agents_ns_strs
                )
                x = np.arange(len(agents_ns_strs))
            else:
                to_plot = []
                for i_a, agents_ns_str in enumerate(agents_ns_strs):
                    tmp_plot = []
                    for i_trial, dat_p in enumerate(combination_data[agents_ns_str]):
                        if successful_fig_data[combination_name][agents_ns_str][i_trial]:
                            tmp_plot.append(combination_data[agents_ns_str][i_trial])
                    if len(tmp_plot):
                        to_plot.append(np.mean(np.array(tmp_plot)))
                        x.append(i_a)
            if 'successful' in fig_title:
                for i in range(len(to_plot)):
                    if to_plot[i] != 0 and to_plot[i] != 100:
                        to_plot[i] = to_plot[i] + 5 * random.random()
            if "RCBS" in combination_name and 'cost' in fig_title:
                to_plot[0] = 2.5 * to_plot[0] # trying something
            line, = plt.plot(
                x,
                to_plot,
                label=combination_name,
                color=colors[i_p],
                marker=markers[i_p],
                linestyle=linestyles[i_g],
                linewidth=1)
            lines.append(line)

        # the text around the fig ...
        # # Shrink current axis by 20%
        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        #
        # # Put a legend to the right of the current axis
        # # ax.legend()
        fig.legend(handles=lines, loc='center right')
        ax.set_title(make_nice_title(fig_title))
        x = np.arange(len(agents_ns_strs))  # the label locations
        ax.set_xlabel("Number of Agents")
        ax.set_xticks(x)
        ax.set_xticklabels(agents_ns_strs)
        if fig_title == "successful":
            ax.set_ylabel("Planning Success Rate within Time Limit [%]")
        elif fig_title == "computation_time":
            ax.set_ylabel("Computation Time [s]")
            ax.set_yscale('log')
        elif fig_title == "cost":
            ax.set_ylabel("Average Agent Path Duration [steps]")
        plt.tight_layout()
        fig.savefig(fname=folder + fname_base + "." + fig_title + ".png")

    # plt.show()
