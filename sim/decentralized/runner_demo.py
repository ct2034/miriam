#!/usr/bin/env python3
import logging
import random

import numpy as np
from definitions import INVALID
from matplotlib import pyplot as plt
from scenarios.evaluators import (cost_ecbs, cost_independent,
                                  cost_sim_decentralized_learned)
from scenarios.generators import random_fill, tracing_pathes_in_the_dark
from scenarios.solvers import ecbs
from scenarios.visualization import plot_with_arrows, plot_with_paths
from sim.decentralized.iterators import IteratorType
from sim.decentralized.policy import PolicyType
from sim.decentralized.runner import run_a_scenario, to_agent_objects

if __name__ == "__main__":  # pragma: no cover
    # a scenario
    env = np.array([[0, 0, 0, 1],
                    [0, 0, 0, 0],
                    [1, 1, 0, 1],
                    [0, 0, 0, 0]])
    starts = np.array([[0, 2],
                       [1, 1],
                       [3, 3]])
    goals = np.array([[0, 1],
                      [3, 1],
                      [1, 1]])
    agents = to_agent_objects(env, starts, goals, rng=random.Random(0))
    # plot_with_arrows(env, starts, goals)
    # plt.show()
    res = run_a_scenario(
        env, agents, False, IteratorType.BLOCKING1,
        ignore_finished_agents=False, print_progress=True)
    print(res)

    # checking big environments
    env, starts, goals = random_fill(40, .2, 50, random.Random(0))
    agents = to_agent_objects(env, starts, goals)
    plot_with_arrows(env, starts, goals)
    plt.show()
    # paths_ecbs = ecbs(env, starts, goals, return_paths=True)
    # plot_with_paths(env, paths_ecbs)
    res = run_a_scenario(
        env, agents, False, IteratorType.BLOCKING1,
        ignore_finished_agents=False, print_progress=True)
    print(res)

    env = np.zeros((3, 3))
    starts = np.array([
        [0, 0],
        [2, 0]
    ])
    goals = np.array([
        [2, 2],
        [0, 2]
    ])
    agents = to_agent_objects(env, starts, goals)
    paths_ecbs = ecbs(env, starts, goals, return_paths=True)
    plot_with_paths(env, paths_ecbs)
    res = run_a_scenario(env, agents, True, IteratorType.BLOCKING3)
    print(res)

    logging.getLogger("sim.decentralized.agent").setLevel(logging.ERROR)
    logging.getLogger("__main__").setLevel(logging.ERROR)
    logging.getLogger("root").setLevel(logging.ERROR)

    size = 8
    n_agents = 8
    policy = PolicyType.LEARNED
    it = IteratorType.BLOCKING3
    interesting = False
    seed = 0

    while(not interesting):
        env, starts, goals = tracing_pathes_in_the_dark(
            size, .5, n_agents, random.Random(seed))
        agents = to_agent_objects(env, starts, goals, policy)
        c_ecbs = cost_ecbs(env, starts, goals)
        c_indep = cost_independent(env, starts, goals)
        c_decen = cost_sim_decentralized_learned(
            env, starts, goals, skip_cache=True)
        print(f"seed: {seed}")
        print(f"c_ecbs: {c_ecbs}, c_indep: {c_indep}, c_decen: {c_decen}")
        if c_ecbs != INVALID and c_indep != INVALID and c_decen != INVALID:
            interesting = c_decen > c_ecbs
        seed += 1

    res_ecbs = ecbs(env, starts, goals, return_paths=False)
    print(res_ecbs['blocks'])
    paths_ecbs = ecbs(env, starts, goals, return_paths=True)
    plot_with_paths(env, paths_ecbs)
    plt.show()

    logging.getLogger("sim.decentralized.policy").setLevel(logging.DEBUG)
    res_decen = run_a_scenario(env, agents, plot=True, iterator=it)
    print(res_decen)
