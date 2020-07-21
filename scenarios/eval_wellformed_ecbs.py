#!/usr/bin/env python3
import logging
import random
import time
from functools import lru_cache
from itertools import product

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scenarios.evaluators import *
from scenarios.generators import like_sim_decentralized

if __name__ == "__main__":
    # no warnings pls
    logging.getLogger('sim.decentralized.agent').setLevel(logging.ERROR)

    size = 16  # size for all scenarios
    n_fills = 8  # how many different fill values there should be
    n_n_agentss = 8  # how many different numbers of agents should there be"""
    n_runs = 4  # how many runs per configuration

    results_well_formed = np.zeros([n_fills, n_n_agentss])  # save results here
    results_ecbs_cost = np.zeros([n_runs, n_fills, n_n_agentss])

    fills = np.around(
        np.linspace(0, .95, n_fills),  # list of fills we want
        2
    )
    # list of different numbers of agents we want
    n_agentss = np.linspace(1, 16, n_n_agentss, dtype=np.int)
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
            results_ecbs_cost[i_r, i_f, i_a] = (
                cost_ecbs(env, starts, goals)
            )
    elapsed_time = time.time() - t
    print("elapsed time: %.3fs" % elapsed_time)

    print(np.min(results_ecbs_cost[results_ecbs_cost != -1]))

    # fix ecbs costs max ............................................
    max_ecbs_cost = np.max(results_ecbs_cost)
    for i_r, i_f, i_a in product(range(n_runs),
                                 range(n_fills), range(n_n_agentss)):
        if results_ecbs_cost[i_r, i_f, i_a] == -1:
            results_ecbs_cost[i_r, i_f, i_a] = max_ecbs_cost
    results_ecbs_cost_final = np.mean(results_ecbs_cost, axis=0)

    # plot well formed ..............................................
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(121)
    im = ax.imshow(
        results_well_formed,
        cmap='plasma',
        origin='lower'
    )
    cbar = fig.colorbar(im, extend='both', spacing='uniform',
                    shrink=0.9, ax=ax)
    plt.title('Well-formedness\n[y: well; b: not well]')
    plt.ylabel('Fills')
    plt.yticks(range(n_fills), map(lambda a: str(a), fills))
    plt.xlabel('Agents')
    plt.xticks(range(n_n_agentss), map(lambda a: str(int(a)), n_agentss))

    # plot ecbs cost ................................................
    ax = fig.add_subplot(122)
    im = ax.imshow(
        results_ecbs_cost_final,
        cmap='plasma',
        origin='lower'
    )
    cbar = fig.colorbar(im, extend='both', spacing='uniform',
                    shrink=0.9, ax=ax)
    plt.title('Ecbs cost\n[y: high; b: low]')
    plt.ylabel('Fills')
    plt.yticks(range(n_fills), map(lambda a: str(a), fills))
    plt.xlabel('Agents')
    plt.xticks(range(n_n_agentss), map(lambda a: str(int(a)), n_agentss))
    plt.show()
