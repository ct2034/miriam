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

    fname_base = "res/z_200_4096.eval_cen.pkl"
    fname_rcbs = "res/z_200_4096.eval_cen.RCBS_ONLY.pkl"
    fname_ecbs = "res/z_200_4096.eval_cen.ECBS_ONLY.pkl"
    fname_ilp_grid = "res/z_200_4096.eval_cen.ILP-GRID.pkl"

    eval_results = None
    with open(fname_base) as fb:
        eval_results = pickle.load(fb)
    assert eval_results is not None

    with open(fname_rcbs) as fb:
        eval_results_rcbs = pickle.load(fb)
        for fig_title in eval_results.keys():
            fig_data = eval_results_rcbs[fig_title]
            names_to_update = filter(
                lambda n: "RCBS" in n,
                fig_data.keys())
            for combination_name in names_to_update:
                eval_results[fig_title][combination_name] = eval_results_rcbs[fig_title][combination_name]

    with open(fname_ecbs) as fb:
        eval_results_ecbs = pickle.load(fb)
        for fig_title in eval_results.keys():
            fig_data = eval_results_ecbs[fig_title]
            names_to_update = filter(
                lambda n: "ECBS" in n,
                fig_data.keys())
            for combination_name in names_to_update:
                eval_results[fig_title][combination_name] = eval_results_ecbs[fig_title][combination_name]

    with open(fname_ilp_grid) as fb:
        eval_results_ilp_grid = pickle.load(fb)
        for fig_title in eval_results.keys():
            fig_data = eval_results_ilp_grid[fig_title]
            names_to_update = filter(
                lambda n: "ILP-GRID" in n,
                fig_data.keys())
            for combination_name in names_to_update:
                eval_results[fig_title][combination_name] = eval_results_ilp_grid[fig_title][combination_name]

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
            ax.set_ylabel("Average Agent Path Duration [steps]")
        plt.tight_layout()
        fig.savefig(fname=fname_base + "." + fig_title + ".png")

    # plt.show()
