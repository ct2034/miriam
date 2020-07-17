#!/usr/bin/env python3
import logging
import random
from functools import lru_cache
from itertools import product
from typing import *

import numpy as np
from matplotlib import pyplot as plt

import sim.decentralized.runner
import sim.decentralized.agent

logging.getLogger('sim.decentralized.agent').setLevel(logging.ERROR)


def like_sim_decentralized(size: int, fill: float,
                           n_agents: int, seed: Any):
    random.seed(seed)
    env = sim.decentralized.runner.initialize_environment(size, fill)
    agents = sim.decentralized.runner.initialize_agents(
        env, n_agents, sim.decentralized.agent.Policy.RANDOM)
    return env, agents
