#!/usr/bin/env python3
import logging
import pickle
import time
from copy import copy
from functools import lru_cache
from itertools import product
from typing import *

import matplotlib.colors as colors
import numpy as np
import pandas as pd
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
DIFFERENCE_ECBS_EN_MINUS_ICTS_EN = "difference_ecbs_en_-_icts_en"


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


def get_fname(generator_name, evaluation, extension):
    return ("scenarios/res_" + generator_name + "-" +
            evaluation + "." + extension)


def genstr(generator):
    generator_name = str(generator).split("at 0x")[
        0].replace("<function ", "").replace(" ", "")
    return generator_name


def plot_results(
        results: List[np.ndarray], titles: List[str],
        generator_name: Callable, n_agentss: List[int], fills: List[float],
        evaluation: str):
    n_fills = len(fills)
    n_n_agentss = len(n_agentss)

    # our cmap with support for over / under
    palette = copy(plt.cm.plasma)
    palette.set_over('w', 1.0)
    palette.set_under('k', 1.0)
    palette.set_bad('r', 1.0)

    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(generator_name, fontsize=16)
    subplot_basenr = 201 + int(np.ceil(len(results) / 2)) * 10
    for i, r in enumerate(results):
        # do we have any results?
        assert not np.all(r == 0)
        assert not np.all(r == -1)
        if len(r.shape) == 3:
            r_final = np.full([n_fills, n_n_agentss], INVALID, dtype=np.float)
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
        if "Difference" not in titles[i]:  # not on the difference
            assert r_min >= 0, "no negative results (except INVALID)"
        r_max = np.max(r_final[r_final != INVALID])
        ax = fig.add_subplot(subplot_basenr+i)
        im = ax.imshow(
            r_final,
            cmap=palette,
            norm=colors.Normalize(vmin=r_min, vmax=r_max),
            origin='lower'
        )
        fig.colorbar(im, extend='both', spacing='uniform',
                     shrink=0.9, ax=ax)
        plt.title(titles[i])
        plt.ylabel('Fills')
        plt.yticks(range(n_fills), map(lambda a: str(a), fills))
        plt.xlabel('Agents')
        plt.xticks(range(n_n_agentss), map(
            lambda a: str(int(a)), n_agentss))
    plt.tight_layout()
    fname = get_fname(generator_name, evaluation, "png")
    plt.savefig(fname)


def main_icts():
    # no warnings pls
    logging.getLogger('sim.decentralized.agent').setLevel(logging.ERROR)

    max_fill, n_fills, n_n_agentss, n_runs, size = init_values_debug()
    # max_fill, n_fills, n_n_agentss, n_runs, size = init_values_main()

    # parameters to evaluate against
    generators = [
        # like_policylearn_gen,
        like_sim_decentralized,
        tracing_pathes_in_the_dark
    ]

    fills = np.around(
        np.linspace(0, max_fill, n_fills),
        2
    )  # list of fills we want

    n_agentss = np.linspace(1, 16, n_n_agentss, dtype=np.int
                            )  # list of different numbers of agents we want

    evaluations = [
        ECBS_SUCCESS,
        ECBS_COST,
        ECBS_EXPANDED_NODES,
        ECBS_VERTEX_BLOCKS,
        ECBS_EDGE_BLOCKS,
        ICTS_SUCCESS,
        ICTS_EXPANDED_NODES,
        DIFFERENCE_ECBS_EN_MINUS_ICTS_EN
    ]

    # preparing panda dataframes
    index_arrays = {
        'generators': list(map(genstr, generators)),
        'fills': fills,
        'n_agentss': n_agentss,
        'evaluations': evaluations
    }
    idx = pd.MultiIndex.from_product(
        index_arrays.values(), names=index_arrays.keys())
    all_results = pd.DataFrame(index=idx)

    all_results.sort_index(inplace=True)
    assert len(index_arrays) == all_results.index.lexsort_depth

    t = time.time()
    i = 0
    for gen, i_r, i_f, i_a in product(generators,
                                      range(n_runs),
                                      range(len(fills)),
                                      range(len(n_agentss))):
        i += 1
        print("run %d of %d" %
              (i, n_runs * n_fills * n_n_agentss * len(generators)))
        # Taking care of some little pandas ...........................
        col = str(i_r)
        if col not in all_results.columns:
            all_results[col] = [-1] * len(all_results.index)
        all_results.sort_index(inplace=True)
        # generating a scenario .......................................
        fill = fills[i_f]
        n_agents = n_agentss[i_a]
        env, starts, goals = gen(
            size, fill, n_agents, seed=i_r)
        # calculating optimal cost ....................................
        if i_f > 0 and all_results.loc[
                (genstr(gen), fills[i_f-1], n_agents, ECBS_COST), col
        ] == INVALID:
            # previous fills timed out
            res_ecbs = INVALID
        elif i_a > 0 and all_results.loc[
                (genstr(gen), fill, n_agentss[i_a-1], ECBS_COST), col
        ] == INVALID:
            # previous agent count failed as well
            res_ecbs = INVALID
        else:
            res_ecbs = cost_ecbs(env, starts, goals)
        if res_ecbs != INVALID:
            all_results.loc[
                (genstr(gen), fill, n_agents, ECBS_SUCCESS), col] = 1
            all_results.loc[
                (genstr(gen), fill, n_agents, ECBS_COST), col] = res_ecbs
            all_results.loc[
                (genstr(gen), fill, n_agents, ECBS_EXPANDED_NODES), col
            ] = expanded_nodes_ecbs(
                env, starts, goals)
            # evaluating blocks
            blocks = blocks_ecbs(env, starts, goals)
            if blocks != INVALID:
                (
                    all_results.loc[
                        (genstr(gen), fill, n_agents,
                         ECBS_VERTEX_BLOCKS), col],
                    all_results.loc[
                        (genstr(gen), fill, n_agents,
                         ECBS_EDGE_BLOCKS), col]
                ) = blocks
            else:
                all_results.loc[
                    (genstr(gen), fill, n_agents, ECBS_VERTEX_BLOCKS), col
                ] = INVALID
                all_results.loc[
                    (genstr(gen), fill, n_agents, ECBS_EDGE_BLOCKS)
                ] = INVALID
        # what is icts cost? ..........................................
        if i_f > 0 and all_results.loc[
                (genstr(gen), fills[i_f-1], n_agents, ICTS_SUCCESS), col
        ] == INVALID:
            # previous fills timed out
            res_icts = INVALID
        elif i_a > 0 and all_results.loc[
                (genstr(gen), fill, n_agentss[i_a-1], ICTS_SUCCESS), col
        ] == INVALID:
            # previous agent count failed as well
            res_icts = INVALID
        else:
            res_icts = cost_icts(env, starts, goals)
        if res_icts != INVALID:
            all_results.loc[
                (genstr(gen), fill, n_agents, ICTS_SUCCESS), col] = 1
            # all_results.loc[
            #     (genstr(gen), fill, n_agents, ICTS_COST), col] = res_icts
            all_results.loc[
                (genstr(gen), fill, n_agents, ICTS_EXPANDED_NODES), col
            ] = expanded_nodes_icts(
                env, starts, goals)
        # run icts and compare n of expanded nodes
        if (
            all_results.loc[
                (genstr(gen), fill, n_agents, ECBS_EXPANDED_NODES), col
            ] != INVALID and
            all_results.loc[
                (genstr(gen), fill, n_agents, ICTS_EXPANDED_NODES), col
            ] != INVALID
        ):
            all_results.loc[
                (genstr(gen), fill, n_agents,
                 DIFFERENCE_ECBS_EN_MINUS_ICTS_EN),
                col
            ] = float(
                all_results.loc[
                    (genstr(gen), fill, n_agents, ECBS_EXPANDED_NODES),
                    col
                ] -
                all_results.loc[
                    (genstr(gen), fill, n_agents, ICTS_EXPANDED_NODES),
                    col
                ]
            )

    with pd.option_context('display.max_rows',
                           None,
                           'display.max_columns',
                           None):  # all rows and columns
        print(all_results)

    elapsed_time = time.time() - t
    print("elapsed time: %.3fs" % elapsed_time)

    for gen in generators:
        results = all_results[str(gen)]
        # saving
        with open(get_fname(genstr(gen), "icts", "pkl"), 'wb') as f:
            pickle.dump(results, f)
        # plot
        plot_results(
            [all_results.loc[ECBS_SUCCESS],
             all_results.loc[ECBS_COST],
             all_results.loc[ECBS_EXPANDED_NODES],
             all_results.loc[ECBS_VERTEX_BLOCKS],
             all_results.loc[ICTS_SUCCESS],
             all_results.loc[ICTS_COST],
             all_results.loc[ICTS_EXPANDED_NODES],
             all_results.loc[DIFFERENCE_ECBS_EN_MINUS_ICTS_EN]],
            ["ECBS success",
             "ECBS cost",
             "ECBS expanded nodes",
             "Nr of vertex blocks in ecbs solution",
             "ICTS success",
             "ICTS cost",
             "ICTS expanded nodes",
             "Difference ECBS minus ICTS expanded nodes"],
            generator_name=genstr(gen),
            n_agentss=n_agentss,
            fills=fills,
            evaluation="icts"
        )
    plt.show()


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
        results[WELL_FORMED] = np.zeros([n_fills, n_n_agentss])
        results[DIFF_INDEP] = np.full(
            [n_runs, n_fills, n_n_agentss], INVALID, dtype=np.float)
        results[DIFF_SIM_DECEN] = np.full(
            [n_runs, n_fills, n_n_agentss], INVALID, dtype=np.float)
        results[ECBS_SUCCESS] = np.zeros([n_fills, n_n_agentss])
        # second plot row
        results[ECBS_COST] = np.full(
            [n_runs, n_fills, n_n_agentss], INVALID, dtype=np.float)
        results[ECBS_VERTEX_BLOCKS] = np.full(
            [n_runs, n_fills, n_n_agentss], INVALID, dtype=np.float)
        results[ECBS_EDGE_BLOCKS] = np.full(
            [n_runs, n_fills, n_n_agentss], INVALID, dtype=np.float)
        results[USEFULLNESS] = np.zeros(
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
                results[WELL_FORMED][i_f, i_a] += is_wellformed
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
                # is this different to the independant costs? .................
                if results[ECBS_COST][i_r, i_f, i_a] != INVALID:
                    cost_indep = cost_independent(env, starts, goals)
                    if cost_indep != INVALID:
                        results[DIFF_INDEP][i_r, i_f, i_a] = (
                            results[ECBS_COST][i_r, i_f, i_a] - cost_indep
                        )
                        results[USEFULLNESS][i_r, i_f,
                                             i_a] += results[
                            DIFF_INDEP][i_r, i_f, i_a]
                # how bad are the costs with sim decentralized random .........
                if results[ECBS_COST][i_r, i_f, i_a] != INVALID:
                    cost_decen = cost_sim_decentralized_random(
                        env, starts, goals)
                    if cost_decen != INVALID:
                        results[DIFF_SIM_DECEN][i_r, i_f, i_a] = (
                            cost_decen - results[ECBS_COST][i_r, i_f, i_a]
                        )
                        results[USEFULLNESS][i_r, i_f,
                                             i_a] += results[
                            DIFF_SIM_DECEN][
                            i_r, i_f, i_a]
    elapsed_time = time.time() - t
    print("elapsed time: %.3fs" % elapsed_time)

    for gen in generators:
        results = all_results[str(gen)]
        # saving
        with open(get_fname(genstr(gen), "main", "pkl"), 'wb') as f:
            pickle.dump(results, f)
        # plot
        plot_results(
            [results[WELL_FORMED],
             results[DIFF_SIM_DECEN],
             results[DIFF_INDEP],
             results[ECBS_SUCCESS],
             results[ECBS_COST],
             results[ECBS_VERTEX_BLOCKS],
             results[ECBS_EDGE_BLOCKS],
             results[USEFULLNESS]],
            ["Well-formedness",
             "Sub-optimality of sim_decen (p: random)",
             "Cost difference ecbs to independant",
             "Ecbs success",
             "Ecbs cost",
             "Nr of vertex blocks in ecbs solution",
             "Nr of edge blocks in ecbs solution",
             "Usefullness"],
            generator_name=genstr(gen),
            n_agentss=n_agentss,
            fills=fills,
            evaluation='main'
        )
    plt.show()


if __name__ == "__main__":
    main_icts()
