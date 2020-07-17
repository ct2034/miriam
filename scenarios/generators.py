#!/usr/bin/env python3
from functools import lru_cache
from itertools import product

import numpy as np
from matplotlib import pyplot as plt

import sim.decentralized


def generate_like_sim_decentralized(size: int, fill: float, n_agents: int):
    env = sim.decentralized.runner.initialize_environment(size, fill)
    agents = sim.decentralized.runner.initialize_agents(
        env, n_agents, sim.decentralized.agent.Policy.RANDOM)
    return env, agents
