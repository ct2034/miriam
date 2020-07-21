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
    n_runs = 2  # how many runs per configuration

    results_well_formed = np.zeros([n_fills, n_n_agentss])  # save results here
    results_ecbs_cost = np.zeros([n_fills, n_n_agentss])  # save results here

    fills = np.around(
        np.linspace(0, .95, n_fills),  # list of fills we want
        2
    )
    # list of different numbers of agents we want
    n_agentss = np.linspace(1, 16, n_n_agentss, dtype=np.int)
    t = time.process_time()
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
                print(
                    cost_ecbs(env, starts, goals)
                )
            except AssertionError:
                is_wellformed = False
            results_well_formed[i_f, i_a] += is_wellformed
    elapsed_time = time.process_time() - t
    print("elapsed time: %.3fs" % elapsed_time)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(
        results_well_formed,
        cmap='plasma',
        origin='lower'
    )
    plt.title('Well-formedness [y: well; b: not well]')
    plt.ylabel('Fills')
    plt.yticks(range(n_fills), map(lambda a: str(a), fills))
    plt.xlabel('Agents')
    plt.xticks(range(n_n_agentss), map(lambda a: str(int(a)), n_agentss))
    plt.show()
