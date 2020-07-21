import logging
from functools import lru_cache

import numpy as np
from cachier import cachier

import tools
from sim.decentralized.agent import Agent, Policy
from sim.decentralized.runner import is_environment_well_formed
from planner.policylearn.libMultiRobotPlanning.plan_ecbs import plan_in_gridmap

logging.getLogger('sim.decentralized.agent').setLevel(logging.ERROR)


@cachier(hash_params=tools.hasher)
def is_well_formed(env, starts, goals):
    n_agents = starts.shape[0]
    agents = []
    for i_a in range(n_agents):
        a = Agent(env, starts[i_a])
        a.give_a_goal(goals[i_a])
        agents.append(a)
    return is_environment_well_formed(tuple(agents))


@cachier(hash_params=tools.hasher)
def cost_ecbs(env, starts, goals):
    """get the average agent cost of this from ecbs

    returns: `float` and `-1` if planning was unsuccessful."""
    n_agents = starts.shape[0]
    data = plan_in_gridmap(env, list(starts), list(goals), 10)
    if data is None:
        return -1
    schedule = data['schedule']
    if n_agents >= len(schedule.keys()):  # Not plans for all agents
        return -1
    cost_per_agent = []
    for i_a in range(n_agents):
        agent_key = 'agent'+str(i_a)
        assert agent_key in schedule.keys(), "Path for this agent"
        path = schedule[agent_key]
        cost_per_agent.append(path[-1]['t'])
    return sum(cost_per_agent) / n_agents
