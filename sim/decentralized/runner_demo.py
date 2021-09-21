#!/usr/bin/env python3
import logging

from definitions import INVALID
from matplotlib import pyplot as plt
from scenarios.evaluators import (cost_ecbs, cost_independent,
                                  cost_sim_decentralized_learned)
from scenarios.generators import tracing_pathes_in_the_dark
from scenarios.solvers import ecbs
from scenarios.visualization import plot_with_paths
from sim.decentralized.iterators import IteratorType
from sim.decentralized.policy import PolicyType
from sim.decentralized.runner import run_a_scenario, to_agent_objects

if __name__ == "__main__":  # pragma: no cover
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
            size, .5, n_agents, seed)
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
