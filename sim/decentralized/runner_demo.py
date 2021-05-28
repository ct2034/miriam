#!/usr/bin/env python3
import logging

from sim.decentralized.iterators import IteratorType
from sim.decentralized.policy import PolicyType
from sim.decentralized.runner import *

if __name__ == "__main__":  # pragma: no cover
    logging.getLogger("sim.decentralized.agent").setLevel(logging.ERROR)
    logging.getLogger("__main__").setLevel(logging.ERROR)
    logging.getLogger("root").setLevel(logging.ERROR)

    size = 4
    n_agents = 4
    policy = PolicyType.LEARNED
    it = IteratorType.BLOCKING3
    sample_and_run_a_scenario(
        size, n_agents, policy, True, 0, it)
