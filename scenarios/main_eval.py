#!/usr/bin/env python3
import logging
import random
import time
from copy import copy
from functools import lru_cache
from itertools import product
from typing import *

import matplotlib.colors as colors
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scenarios.evaluators import *
from scenarios.generators import *


WELL_FORMED = "well_formed"
DIFF_INDEP = "diff_indep"
DIFF_SIM_DECEN = "diff_sim_decen"
ECBS_SUCCESS = "ecbs_success"
ECBS_COST = "ecbs_cost"
ECBS_VERTEX_BLOCKS = "ecbs_vertex_blocks"
ECBS_EDGE_BLOCKS = "ecbs_edge_blocks"
ECBS_EXPANDED_NODES = "ecbs_expanded_nodes"
USEFULLNESS = "usefullness"
ICTS_SUCCESS = "icts_success"
ICTS_COST = "icts_cost"
ICTS_EXPANDED_NODES = "icts_expanded_nodes"
QUOTIENT_ECBS_EN_OVER_ICTS_EN = "quotient_ecbs_en_over_icts_en"


def init_values_debug():
    size = 8  # size for all scenarios
    n_fills = 2  # how many different fill values there should be
    n_n_agentss = 2  # how many different numbers of agents should there be"""
    n_runs = 2  # how many runs per configuration
    max_fill = .4  # maximal fill to sample until
    return max_fill, n_fills, n_n_agentss, n_runs, size


def init_values_main():
    size = 8  # size for all scenarios
    n_fills = 8  # how many different fill values there should be
    n_n_agentss = 8  # how many different numbers of agents should there be"""
    n_runs = 16  # how many runs per configuration
    max_fill = .6  # maximal fill to sample until
    return max_fill, n_fills, n_n_agentss, n_runs, size


def plot_results(
        results: List[np.ndarray], titles: List[str],
        generator: Callable, n_agentss: List[int], fills: List[float],
        evaluation: str = 'main'):
    n_fills = len(fills)
    n_n_agentss = len(n_agentss)

    # our cmap with support for over / under
    palette = copy(plt.cm.plasma)
    palette.set_over('w', 1.0)
    palette.set_under('k', 1.0)
    palette.set_bad('r', 1.0)

    fig = plt.figure(figsize=(20, 10))
    generator_name = str(generator).split("at 0x")[
        0].replace("<function ", "").replace(" ", "")
    fig.suptitle(generator_name, fontsize=16)
    subplot_basenr = 201 + int(np.ceil(len(results) / 2)) * 10
    for i, r in enumerate(results):
        # do we have any results?
        assert not np.all(r == 0)
        assert not np.all(r == -1)
        if len(r.shape) == 3:
            r_final = np.full([n_fills, n_n_agentss], INVALID)
            for i_f, i_a in product(range(n_fills),
                                    range(n_n_agentss)):
                this_data = r[:, i_f, i_a]
                if len(this_data[this_data != INVALID]) > 0:
                    r_final[i_f, i_a] = np.mean(
                        this_data[this_data != INVALID]
                    )
        elif len(r.shape) == 2:
            r_final = r
        else:
            raise RuntimeError("results must have 2 or 3 dimensions")
        r_min = np.min(r_final[r_final != INVALID])
        assert r_min >= 0, "no negative results (except INVALID)"
        r_max = np.max(r_final[r_final != INVALID])
        ax = fig.add_subplot(subplot_basenr+i)
        im = ax.imshow(
            r_final,
            cmap=palette,
            norm=colors.Normalize(vmin=r_min, vmax=r_max),
            origin='lower'
        )
        cbar = fig.colorbar(im, extend='both', spacing='uniform',
                            shrink=0.9, ax=ax)
        plt.title(titles[i])
        plt.ylabel('Fills')
        plt.yticks(range(n_fills), map(lambda a: str(a), fills))
        plt.xlabel('Agents')
        plt.xticks(range(n_n_agentss), map(
            lambda a: str(int(a)), n_agentss))
    plt.tight_layout()
    plt.savefig("scenarios/res_" + generator_name + "-" + evaluation + ".png")


def main_icts():
    # no warnings pls
    logging.getLogger('sim.decentralized.agent').setLevel(logging.ERROR)

    max_fill, n_fills, n_n_agentss, n_runs, size = init_values_debug()
    # max_fill, n_fills, n_n_agentss, n_runs, size = init_values_main()

    generators = [
        like_policylearn_gen,
        like_sim_decentralized,
        tracing_pathes_in_the_dark
    ]
    all_results = {}

    fills = np.around(
        np.linspace(0, max_fill, n_fills),
        2
    )  # list of fills we want

    n_agentss = np.linspace(1, 16, n_n_agentss, dtype=np.int
                            )  # list of different numbers of agents we want

    for gen in generators:
        all_results[str(gen)] = {}
        results = all_results[str(gen)]

        # save results here
        # first plot row
        results[ECBS_SUCCESS] = np.zeros([n_fills, n_n_agentss])
        results[ECBS_COST] = np.full(
            [n_runs, n_fills, n_n_agentss], INVALID, dtype=np.float)
        results[ECBS_EXPANDED_NODES] = np.full(
            [n_runs, n_fills, n_n_agentss], INVALID, dtype=np.float)
        results[ECBS_VERTEX_BLOCKS] = np.full(
            [n_runs, n_fills, n_n_agentss], INVALID, dtype=np.float)
        results[ECBS_EDGE_BLOCKS] = np.full(  # don't plot (maybe)
            [n_runs, n_fills, n_n_agentss], INVALID, dtype=np.float)
        # second plot row
        results[ICTS_SUCCESS] = np.zeros([n_fills, n_n_agentss])
        results[ICTS_COST] = np.full(
            [n_runs, n_fills, n_n_agentss], INVALID, dtype=np.float)
        results[ICTS_EXPANDED_NODES] = np.full(
            [n_runs, n_fills, n_n_agentss], INVALID, dtype=np.float)
        results[QUOTIENT_ECBS_EN_OVER_ICTS_EN] = np.full(
            [n_runs, n_fills, n_n_agentss], INVALID, dtype=np.float)

    t = time.time()
    i = 0
    for gen in generators:
        results = all_results[str(gen)]
        for i_r in range(n_runs):
            for i_f, i_a in product(range(n_fills),
                                    range(n_n_agentss)):
                i += 1
                print("run %d of %d" %
                      (i, n_runs * n_fills * n_n_agentss * len(generators)))
                # generating a scenario .......................................
                fill = fills[i_f]
                n_agents = n_agentss[i_a]
                env, starts, goals = gen(
                    size, fill, n_agents, seed=i_r)
                # calculating optimal cost ....................................
                if i_f > 0 and results[ECBS_COST
                                       ][i_r, i_f - 1, i_a] == INVALID:
                    # previous fills timed out
                    res_ecbs = INVALID
                elif i_a > 0 and results[ECBS_COST
                                         ][i_r, i_f, i_a - 1] == INVALID:
                    # previous agent count failed as well
                    res_ecbs = INVALID
                else:
                    res_ecbs = cost_ecbs(env, starts, goals)
                if res_ecbs != INVALID:
                    results[ECBS_SUCCESS][i_f, i_a] += 1
                    results[ECBS_COST][i_r, i_f, i_a] = res_ecbs
                    results[ECBS_EXPANDED_NODES
                            ][i_r, i_f, i_a] = expanded_nodes_ecbs(
                                env, starts, goals)
                # evaluating blocks
                blocks = blocks_ecbs(env, starts, goals)
                if blocks != INVALID:
                    (
                        results[ECBS_VERTEX_BLOCKS][i_r, i_f, i_a],
                        results[ECBS_EDGE_BLOCKS][i_r, i_f, i_a]
                    ) = blocks
                else:
                    results[ECBS_VERTEX_BLOCKS][i_r, i_f, i_a] = INVALID
                    results[ECBS_EDGE_BLOCKS][i_r, i_f, i_a] = INVALID
                # what is icts cost? ..........................................
                if i_f > 0 and results[ICTS_COST
                                       ][i_r, i_f - 1, i_a] == INVALID:
                    # previous fills timed out
                    res_icts = INVALID
                elif i_a > 0 and results[ICTS_COST
                                         ][i_r, i_f, i_a - 1] == INVALID:
                    # previous agent count failed as well
                    res_icts = INVALID
                else:
                    res_icts = cost_icts(env, starts, goals)
                if res_icts != INVALID:
                    results[ICTS_SUCCESS][i_f, i_a] += 1
                    results[ICTS_COST][i_r, i_f, i_a] = res_ecbs
                    results[ICTS_EXPANDED_NODES
                            ][i_r, i_f, i_a] = expanded_nodes_icts(
                                env, starts, goals)
                # run icts and compare n of expanded nodes
                if (
                    results[ECBS_EXPANDED_NODES][i_r, i_f, i_a] != INVALID and
                    results[ICTS_EXPANDED_NODES][i_r, i_f, i_a] != INVALID
                ):
                    if (results[ECBS_EXPANDED_NODES][i_r, i_f, i_a] == 0 and
                            results[ICTS_EXPANDED_NODES][i_r, i_f, i_a] == 0):
                        q = 0
                    elif results[ICTS_EXPANDED_NODES][i_r, i_f, i_a] == 0:
                        q = 1000
                    else:
                        q = (
                            results[ECBS_EXPANDED_NODES][i_r, i_f, i_a] /
                            results[ICTS_EXPANDED_NODES][i_r, i_f, i_a]
                        )
                    results[QUOTIENT_ECBS_EN_OVER_ICTS_EN][i_r, i_f, i_a] = q

    elapsed_time = time.time() - t
    print("elapsed time: %.3fs" % elapsed_time)

    # for gen in generators:
    #     results = all_results[str(gen)]
    #     plot_results(
    #         [results[WELL_FORMED],
    #          results[DIFF_SIM_DECEN],
    #          results[DIFF_INDEP],
    #          results[ECBS_SUCCESS],
    #          results[ECBS_COST],
    #          results[ECBS_VERTEX_BLOCKS],
    #          results[ECBS_EDGE_BLOCKS],
    #          results[USEFULLNESS]],
    #         ["Well-formedness",
    #          "Sub-optimality of sim_decen (p: random)",
    #          "Cost difference ecbs to independant",
    #          "Ecbs success",
    #          "Ecbs cost",
    #          "Nr of vertex blocks in ecbs solution",
    #          "Nr of edge blocks in ecbs solution",
    #          "Usefullness"],
    #         generator=gen,
    #         n_agentss=n_agentss,
    #         fills=fills,
    #         evaluation="icts"
    #     )
    # plt.show()


def main_base():
    # no warnings pls
    logging.getLogger('sim.decentralized.agent').setLevel(logging.ERROR)

    max_fill, n_fills, n_n_agentss, n_runs, size = init_values_main()
    generators = [
        like_policylearn_gen,
        like_sim_decentralized,
        tracing_pathes_in_the_dark
    ]
    all_results = {}

    fills = np.around(
        np.linspace(0, max_fill, n_fills),
        2
    )  # list of fills we want

    n_agentss = np.linspace(1, 16, n_n_agentss, dtype=np.int
                            )  # list of different numbers of agents we want

    for gen in generators:
        all_results[str(gen)] = {}
        results = all_results[str(gen)]

        # save results here
        # first plot row
        results["well_formed"] = np.zeros([n_fills, n_n_agentss])
        results["diff_indep"] = np.full(
            [n_runs, n_fills, n_n_agentss], INVALID, dtype=np.float)
        results["diff_sim_decen"] = np.full(
            [n_runs, n_fills, n_n_agentss], INVALID, dtype=np.float)
        results["ecbs_success"] = np.zeros([n_fills, n_n_agentss])
        # second plot row
        results["ecbs_cost"] = np.full(
            [n_runs, n_fills, n_n_agentss], INVALID, dtype=np.float)
        results["ecbs_vertex_blocks"] = np.full(
            [n_runs, n_fills, n_n_agentss], INVALID, dtype=np.float)
        results["ecbs_edge_blocks"] = np.full(
            [n_runs, n_fills, n_n_agentss], INVALID, dtype=np.float)
        results["usefullness"] = np.zeros(
            [n_runs, n_fills, n_n_agentss], dtype=np.float)

    t = time.time()
    i = 0
    for gen in generators:
        results = all_results[str(gen)]
        for i_r in range(n_runs):
            for i_f, i_a in product(range(n_fills),
                                    range(n_n_agentss)):
                i += 1
                print("run %d of %d" %
                      (i, n_runs * n_fills * n_n_agentss * len(generators)))
                # generating a scenario .......................................
                fill = fills[i_f]
                n_agents = n_agentss[i_a]
                try:
                    env, starts, goals = gen(
                        size, fill, n_agents, seed=i_r)
                    is_wellformed = (
                        is_well_formed(
                            env, starts, goals))
                except AssertionError:
                    is_wellformed = False
                results["well_formed"][i_f, i_a] += is_wellformed
                # calculating optimal cost ....................................
                if i_f > 0 and results["ecbs_cost"
                                       ][i_r, i_f - 1, i_a] == INVALID:
                    # previous fills timed out
                    res_ecbs = INVALID
                elif i_a > 0 and results["ecbs_cost"
                                         ][i_r, i_f, i_a - 1] == INVALID:
                    # previous agent count failed as well
                    res_ecbs = INVALID
                else:
                    res_ecbs = cost_ecbs(env, starts, goals)
                if res_ecbs != INVALID:
                    results["ecbs_success"][i_f, i_a] += 1
                    results["ecbs_cost"][i_r, i_f, i_a] = res_ecbs
                # evaluating blocks
                blocks = blocks_ecbs(env, starts, goals)
                if blocks != INVALID:
                    (
                        results["ecbs_vertex_blocks"][i_r, i_f, i_a],
                        results["ecbs_edge_blocks"][i_r, i_f, i_a]
                    ) = blocks
                else:
                    results["ecbs_vertex_blocks"][i_r, i_f, i_a] = INVALID
                    results["ecbs_edge_blocks"][i_r, i_f, i_a] = INVALID
                # is this different to the independant costs? .................
                if results["ecbs_cost"][i_r, i_f, i_a] != INVALID:
                    cost_indep = cost_independant(env, starts, goals)
                    if cost_indep != INVALID:
                        results["diff_indep"][i_r, i_f, i_a] = (
                            results["ecbs_cost"][i_r, i_f, i_a] - cost_indep
                        )
                        results["usefullness"][i_r, i_f,
                                               i_a] += results[
                                                   "diff_indep"][i_r, i_f, i_a]
                # how bad are the costs with sim decentralized random .........
                if results["ecbs_cost"][i_r, i_f, i_a] != INVALID:
                    cost_decen = cost_sim_decentralized_random(
                        env, starts, goals)
                    if cost_decen != INVALID:
                        results["diff_sim_decen"][i_r, i_f, i_a] = (
                            cost_decen - results["ecbs_cost"][i_r, i_f, i_a]
                        )
                        results["usefullness"][i_r, i_f,
                                               i_a] += results[
                                                   "diff_sim_decen"][
                                                       i_r, i_f, i_a]
    elapsed_time = time.time() - t
    print("elapsed time: %.3fs" % elapsed_time)

    for gen in generators:
        results = all_results[str(gen)]
        plot_results(
            [results["well_formed"],
             results["diff_sim_decen"],
             results["diff_indep"],
             results["ecbs_success"],
             results["ecbs_cost"],
             results["ecbs_vertex_blocks"],
             results["ecbs_edge_blocks"],
             results["usefullness"]],
            ["Well-formedness",
             "Sub-optimality of sim_decen (p: random)",
             "Cost difference ecbs to independant",
             "Ecbs success",
             "Ecbs cost",
             "Nr of vertex blocks in ecbs solution",
             "Nr of edge blocks in ecbs solution",
             "Usefullness"],
            generator=gen,
            n_agentss=n_agentss,
            fills=fills
        )
    plt.show()


if __name__ == "__main__":
    main_icts()
