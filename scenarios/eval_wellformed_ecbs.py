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

if __name__ == "__main__":
    # no warnings pls
    logging.getLogger('sim.decentralized.agent').setLevel(logging.ERROR)

    size = 8  # size for all scenarios
    n_fills = 8  # how many different fill values there should be
    n_n_agentss = 8  # how many different numbers of agents should there be"""
    n_runs = 2  # how many runs per configuration
    max_fill = .6  # maximal fill to sample until

    results_well_formed = np.zeros([n_fills, n_n_agentss])  # save results here
    results_ecbs_cost = np.zeros([n_runs, n_fills, n_n_agentss])
    results_ecbs_cost.fill(INVALID)
    results_diff_indep = np.zeros([n_runs, n_fills, n_n_agentss])
    results_diff_indep.fill(INVALID)

    fills = np.around(
        np.linspace(0, max_fill, n_fills),
        2
    )  # list of fills we want

    n_agentss = np.linspace(1, 16, n_n_agentss, dtype=np.int
                            )  # list of different numbers of agents we want

    t = time.time()
    for i_r in range(n_runs):
        for i_f, i_a in product(range(n_fills),
                                range(n_n_agentss)):
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
            results_ecbs_cost[i_r, i_f, i_a] = (
                cost_ecbs(env, starts, goals)
            )
            # is this different to the independant costs? .....................
            if results_ecbs_cost[i_r, i_f, i_a] != INVALID:
                cost_indep = cost_independant(env, starts, goals)
                results_diff_indep[i_r, i_f, i_a] = (
                    results_ecbs_cost[i_r, i_f, i_a] - cost_indep) * n_agents
    elapsed_time = time.time() - t
    print("elapsed time: %.3fs" % elapsed_time)

    # do we have any results?
    assert not np.all(results_well_formed == 0)
    assert not np.all(results_ecbs_cost == -1)
    assert not np.all(results_diff_indep == -1)

    # our cmap with support for over / under
    palette = copy(plt.cm.plasma)
    palette.set_over('w', 1.0)
    palette.set_under('k', 1.0)
    palette.set_bad('r', 1.0)

    # plot well formed ........................................................
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(131)
    im = ax.imshow(
        results_well_formed,
        cmap=palette,
        norm=colors.Normalize(vmin=1, vmax=n_runs),
        origin='lower'
    )
    cbar = fig.colorbar(im, extend='both', spacing='uniform',
                        shrink=0.9, ax=ax)
    plt.title('Well-formedness')
    plt.ylabel('Fills')
    plt.yticks(range(n_fills), map(lambda a: str(a), fills))
    plt.xlabel('Agents')
    plt.xticks(range(n_n_agentss), map(lambda a: str(int(a)), n_agentss))

    # fix ecbs costs mean .....................................................
    results_ecbs_cost_final = np.zeros([n_fills, n_n_agentss])
    for i_f, i_a in product(range(n_fills),
                            range(n_n_agentss)):
        this_data = results_ecbs_cost[:, i_f, i_a]
        if len(this_data[this_data != INVALID]) > 0:
            results_ecbs_cost_final[i_f, i_a] = np.mean(
                this_data[this_data != INVALID]
            )
        else:
            results_ecbs_cost_final[i_f, i_a] = INVALID

    # plot ecbs cost ..........................................................
    ax = fig.add_subplot(132)
    ecbs_min = np.min(results_ecbs_cost[results_ecbs_cost != INVALID])
    ecbs_max = np.max(results_ecbs_cost[results_ecbs_cost != INVALID])
    im = ax.imshow(
        results_ecbs_cost_final,
        cmap=palette,
        norm=colors.Normalize(vmin=ecbs_min, vmax=ecbs_max),
        origin='lower'
    )
    cbar = fig.colorbar(im, extend='both', spacing='uniform',
                        shrink=0.9, ax=ax)
    plt.title('Ecbs cost')
    plt.ylabel('Fills')
    plt.yticks(range(n_fills), map(lambda a: str(a), fills))
    plt.xlabel('Agents')
    plt.xticks(range(n_n_agentss), map(lambda a: str(int(a)), n_agentss))

    # difference indep mean ...................................................
    results_diff_indep_final = np.zeros([n_fills, n_n_agentss])
    for i_f, i_a in product(range(n_fills),
                            range(n_n_agentss)):
        this_data = results_diff_indep[:, i_f, i_a]
        if len(this_data[this_data != INVALID]) > 0:
            results_diff_indep_final[i_f, i_a] = np.mean(
                this_data[this_data != INVALID]
            )
        else:
            results_diff_indep_final[i_f, i_a] = INVALID

    # plot cost difference ....................................................
    ax = fig.add_subplot(133)
    diff_indep_min = np.min(results_diff_indep[results_diff_indep != INVALID])
    diff_indep_max = np.max(results_diff_indep[results_diff_indep != INVALID])
    im = ax.imshow(
        results_diff_indep_final,
        cmap=palette,
        norm=colors.Normalize(vmin=diff_indep_min, vmax=diff_indep_max),
        origin='lower'
    )
    cbar = fig.colorbar(im, extend='both', spacing='uniform',
                        shrink=0.9, ax=ax)
    plt.title('Cost difference ecbs to independant')
    plt.ylabel('Fills')
    plt.yticks(range(n_fills), map(lambda a: str(a), fills))
    plt.xlabel('Agents')
    plt.xticks(range(n_n_agentss), map(lambda a: str(int(a)), n_agentss))
    plt.show()
