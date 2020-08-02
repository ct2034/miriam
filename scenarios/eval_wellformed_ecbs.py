#!/usr/bin/env python3
import logging
import random
import time
from copy import copy
from functools import lru_cache
from itertools import product

import matplotlib.colors as colors
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scenarios.evaluators import *
from scenarios.generators import like_sim_decentralized


def plot_results(results, titles):
    # our cmap with support for over / under
    palette = copy(plt.cm.plasma)
    palette.set_over('w', 1.0)
    palette.set_under('k', 1.0)
    palette.set_bad('r', 1.0)

    fig = plt.figure(figsize=(20, 10))
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
    plt.show()


if __name__ == "__main__":
    # no warnings pls
    logging.getLogger('sim.decentralized.agent').setLevel(logging.ERROR)

    size = 8  # size for all scenarios
    n_fills = 8  # how many different fill values there should be
    n_n_agentss = 8  # how many different numbers of agents should there be"""
    n_runs = 10  # how many runs per configuration
    max_fill = .6  # maximal fill to sample until

    # save results here first plot
    results_well_formed = np.zeros([n_fills, n_n_agentss])
    results_diff_indep = np.full(
        [n_runs, n_fills, n_n_agentss], INVALID, dtype=np.float)
    results_diff_sim_decen = np.full(
        [n_runs, n_fills, n_n_agentss], INVALID, dtype=np.float)

    # second plot ecbs
    results_ecbs_success = np.zeros([n_fills, n_n_agentss])
    results_ecbs_cost = np.full(
        [n_runs, n_fills, n_n_agentss], INVALID, dtype=np.float)
    results_ecbs_vertex_blocks = np.full(
        [n_runs, n_fills, n_n_agentss], INVALID, dtype=np.float)
    results_ecbs_edge_blocks = np.full(
        [n_runs, n_fills, n_n_agentss], INVALID, dtype=np.float)

    fills = np.around(
        np.linspace(0, max_fill, n_fills),
        2
    )  # list of fills we want

    n_agentss = np.linspace(1, 16, n_n_agentss, dtype=np.int
                            )  # list of different numbers of agents we want

    t = time.time()
    i = 0
    for i_r in range(n_runs):
        for i_f, i_a in product(range(n_fills),
                                range(n_n_agentss)):
            i += 1
            print("run %d of %d" % (i, n_runs * n_fills * n_n_agentss))
            fill = fills[i_f]
            n_agents = n_agentss[i_a]
            try:
                env, starts, goals = like_sim_decentralized(
                    size, fill, n_agents, seed=i_r)
                is_wellformed = (
                    is_well_formed(
                        env, starts, goals))
            except AssertionError:
                is_wellformed = False
            results_well_formed[i_f, i_a] += is_wellformed
            # calculating optimal cost ........................................
            res_ecbs = cost_ecbs(env, starts, goals)
            results_ecbs_cost[i_r, i_f, i_a] = res_ecbs
            if res_ecbs != INVALID:
                results_ecbs_success[i_f, i_a] += 1
            # evaluating blocks
            blocks = blocks_ecbs(env, starts, goals)
            if blocks != INVALID:
                (
                    results_ecbs_vertex_blocks[i_r, i_f, i_a],
                    results_ecbs_edge_blocks[i_r, i_f, i_a]
                ) = blocks
            else:
                results_ecbs_vertex_blocks[i_r, i_f, i_a] = INVALID
                results_ecbs_edge_blocks[i_r, i_f, i_a] = INVALID
            # is this different to the independant costs? .....................
            if results_ecbs_cost[i_r, i_f, i_a] != INVALID:
                cost_indep = cost_independant(env, starts, goals)
                if cost_indep != INVALID:
                    results_diff_indep[i_r, i_f, i_a] = (
                        results_ecbs_cost[i_r, i_f, i_a] - cost_indep
                    )
            # how bad are the costs with sim decentralized random .............
            if results_ecbs_cost[i_r, i_f, i_a] != INVALID:
                cost_decen = cost_sim_decentralized_random(env, starts, goals)
                if cost_decen != INVALID:
                    results_diff_sim_decen[i_r, i_f, i_a] = (
                        cost_decen - results_ecbs_cost[i_r, i_f, i_a]
                    )
    elapsed_time = time.time() - t
    print("elapsed time: %.3fs" % elapsed_time)

    plot_results(
        [results_well_formed,
         results_diff_sim_decen,
         results_diff_indep,
         results_ecbs_success,
         results_ecbs_cost,
         results_ecbs_vertex_blocks,
         results_ecbs_edge_blocks],
        ["Well-formedness",
         "Sub-optimality of sim_decen (p: random)",
         "Cost difference ecbs to independant",
         "Ecbs success",
         "Ecbs cost",
         "Nr of vertex blocks in ecbs solution",
         "Nr of edge blocks in ecbs solution"]
    )
