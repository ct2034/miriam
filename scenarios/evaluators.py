import logging
from functools import lru_cache

import numpy as np
from cachier import cachier

import tools
from sim.decentralized.agent import Agent, Policy
from sim.decentralized.runner import is_environment_well_formed

logging.getLogger('sim.decentralized.agent').setLevel(logging.ERROR)


# @cachier(hash_params=tools.hasher)
def is_well_formed(env, starts, goals):
    n_agents = starts.shape[0]
    agents = []
    for i_a in range(n_agents):
        a = Agent(env, starts[i_a])
        a.give_a_goal(goals[i_a])
        agents.append(a)
    return is_environment_well_formed(tuple(agents))
