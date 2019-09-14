#!/usr/bin/env python2
import random
import sys
from enum import Enum, unique
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


if __name__ == "__main__":
    plt.style.use('bmh')
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    markers = ['s', 'X', 'o']
    linestyles = ['solid', 'dashed', 'dotted']

    fname = sys.argv[1]
    assert is_eval_cen_file(fname)

    eval_results = None
    with open(fname) as fig:
        eval_results = pickle.load(fig)

    assert eval_results is not None

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
            if fig_title == 'successful':
                to_plot = map(
                    lambda k: 100. * np.count_nonzero(np.array(combination_data[k]))
                              / len(combination_data[k]),
                    agents_ns_strs
                )
            else:
                to_plot = map(
                    lambda k: np.mean(np.array(combination_data[k])),
                    agents_ns_strs
                )
            # to_plot[0] = to_plot[0] + random.random() * 30
            # to_plot[1] = random.random() * to_plot[1] + random.random() * 30
            # to_plot[2] = to_plot[2] + random.random() * 30
            line, = plt.plot(
                to_plot,
                label=combination_name,
                color=colors[i_p],
                marker=markers[i_g],
                linestyle=linestyles[i_g],
                linewidth=1)
            lines.append(line)

        # the text around the fig ...
        fig.legend(handles=lines)
        ax.set_title(make_nice_title(fig_title))
        x = np.arange(len(agents_ns_strs))  # the label locations
        ax.set_xlabel("Number of Agents")
        ax.set_xticks(x)
        ax.set_xticklabels(agents_ns_strs)
        if fig_title == "successful":
            ax.set_ylabel("Planning Success Rate within Time Limit [%]")
        elif fig_title == "computation_time":
            ax.set_ylabel("Computation Time [s]")
        elif fig_title == "cost":
            ax.set_ylabel("Average Agent Path Length [m]")
        plt.tight_layout()
        fig.savefig(fname=fname + "." + fig_title + ".png")

    # plt.show()
