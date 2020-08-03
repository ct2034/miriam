#!/usr/bin/env python3
import logging
import random
from itertools import product
from typing import *

from cachier import cachier
import numpy as np
from matplotlib import pyplot as plt

import sim.decentralized.agent
import sim.decentralized.runner
import tools
import planner.policylearn.generate_data

logging.getLogger('sim.decentralized.agent').setLevel(logging.ERROR)


def make_starts_goals_on_env(env: np.ndarray, n_agents: int):
    agents = sim.decentralized.runner.initialize_agents(
        env, n_agents, sim.decentralized.agent.Policy.RANDOM)
    starts = np.array([a.pos for a in agents])
    assert starts.shape == (n_agents, 2)
    goals = np.array([a.goal for a in agents])
    assert goals.shape == (n_agents, 2)
    return starts, goals


@cachier(hash_params=tools.hasher)
def like_sim_decentralized(size: int, fill: float,
                           n_agents: int, seed: Any):
    random.seed(seed)
    env = sim.decentralized.runner.initialize_environment(size, fill)
    starts, goals = make_starts_goals_on_env(env, n_agents)
    return env, starts, goals


@cachier(hash_params=tools.hasher)
def like_policylearn_gen(size: int, fill: float,
                         n_agents: int, seed: Any):
    random.seed(seed)
    env = planner.policylearn.generate_data.generate_random_gridmap(
        size, size, fill)
    starts, goals = make_starts_goals_on_env(env, n_agents)
    return env, starts, goals
