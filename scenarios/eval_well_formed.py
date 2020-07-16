#!/usr/bin/env python3
import logging
from itertools import product

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sim.decentralized.runner
from scenarios.generators import generate_like_sim_decentralized

if __name__ == "__main__":
    # no warnings pls
    logging.getLogger('sim.decentralized.agent').setLevel(logging.ERROR)  

    size = 10  # size for all scenarios
    n_fills = 20  # how many different fill values there should be
    n_n_agentss = 10  # how many different numbers of agents should there be"""
    n_runs = 10  # how many runs per configuration

    results = np.zeros([n_fills, n_n_agentss])  # save results here

    fills = np.around(
        np.linspace(0, .9, n_fills),  # list of fills we want
        2
    )
    # list of different numbers of agents we want
    n_agentss = np.linspace(1, n_n_agentss, n_n_agentss)

    for _ in range(n_runs):
        for i_f, i_a in product(range(n_fills),
                                range(n_n_agentss)):
            fill = fills[i_f]
            n_agents = n_agentss[i_a]
            try:
                _, agents = generate_like_sim_decentralized(
                    size, fill, int(n_agents))
                is_wellformed = sim.decentralized.runner.is_environment_well_formed(
                    agents)
            except AssertionError:
                is_wellformed = False
            results[i_f, i_a] += is_wellformed

    print("done")
    print(results)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(
        results,
        cmap='plasma',
        origin='lower'
    )
    plt.title('Well-formedness [y: well; b: not well]')
    plt.ylabel('Fills')
    plt.yticks(range(n_fills), map(lambda a: str(a), fills))
    plt.xlabel('Agents')
    plt.xticks(range(n_n_agentss), map(lambda a: str(int(a)), n_agentss))
    plt.show()
