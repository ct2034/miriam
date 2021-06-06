#!/usr/bin/env python3
import logging

from scenarios.evaluators import to_agent_objects
from scenarios.generators import tracing_pathes_in_the_dark
from sim.decentralized.iterators import IteratorType
from sim.decentralized.policy import PolicyType
from sim.decentralized.runner import *

if __name__ == "__main__":  # pragma: no cover
    logging.getLogger("sim.decentralized.agent").setLevel(logging.ERROR)
    logging.getLogger("__main__").setLevel(logging.ERROR)
    logging.getLogger("root").setLevel(logging.ERROR)

    size = 8
    n_agents = 8
    policy = PolicyType.LEARNED
    it = IteratorType.BLOCKING3
    env, starts, goals = tracing_pathes_in_the_dark(size, .5, n_agents, 1)
    # in this scenario, I think agent 3 (? the green one) makes a weird decision.

    agents = to_agent_objects(env, starts, goals, policy)
    res = run_a_scenario(env, agents, plot=True,
                         iterator=it)
    print(res)
