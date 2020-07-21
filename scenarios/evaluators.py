import logging
from functools import lru_cache

import numpy as np
from cachier import cachier

import tools
from sim.decentralized.agent import Agent, Policy
from sim.decentralized.runner import is_environment_well_formed
from planner.policylearn.libMultiRobotPlanning.plan_ecbs import plan_in_gridmap

logging.getLogger('sim.decentralized.agent').setLevel(logging.ERROR)
logging.getLogger(
    'planner.policylearn.libMultiRobotPlanning').setLevel(logging.ERROR)

INVALID = -1


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
        return INVALID
    schedule = data['schedule']
    if n_agents != len(schedule.keys()):  # Not plans for all agents
        return INVALID
    cost_per_agent = []
    for i_a in range(n_agents):
        agent_key = 'agent'+str(i_a)
        assert agent_key in schedule.keys(), "Path for this agent"
        path = schedule[agent_key]
        cost_per_agent.append(path[-1]['t'])
    return sum(cost_per_agent) / n_agents


def cost_independant(env, starts, goals):
    """what would be the average agent cost if the agents would be independent of each other"""
    n_agents = starts.shape[0]
    agents = []
    cost_per_agent = []
    for i_a in range(n_agents):
        a = Agent(env, starts[i_a])
        success = a.give_a_goal(goals[i_a])
        if not success:
            return INVALID
        cost_per_agent.append(len(a.path)-1)
    return sum(cost_per_agent) / n_agents
