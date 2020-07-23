import logging
from functools import lru_cache
from typing import Tuple

import numpy as np
from cachier import cachier

import tools
from planner.policylearn.libMultiRobotPlanning.plan_ecbs import plan_in_gridmap
from sim.decentralized.agent import Agent, Policy
from sim.decentralized.runner import is_environment_well_formed, run_a_scenario

logging.getLogger('sim.decentralized.agent').setLevel(logging.ERROR)
logging.getLogger(
    'planner.policylearn.libMultiRobotPlanning').setLevel(logging.ERROR)

INVALID = -1


def to_agent_objects(env, starts, goals, policy=Policy.RANDOM):
    n_agents = starts.shape[0]
    agents = []
    for i_a in range(n_agents):
        a = Agent(env, starts[i_a])
        a.give_a_goal(goals[i_a])
        agents.append(a)
    return agents


@cachier(hash_params=tools.hasher)
def is_well_formed(env, starts, goals):
    """Check if the environment is well formed according to Cap2015"""
    agents = to_agent_objects(env, starts, goals)
    return is_environment_well_formed(tuple(agents))


@cachier(hash_params=tools.hasher)
def plan_ecbs(env, starts, goals):
    """Plan scenario using ecbs returning results data"""
    try:
        data = plan_in_gridmap(env, list(starts), list(goals), 10)
    except KeyError:  # happens when start or goal is not in map
        return INVALID
    if data is None:
        return INVALID
    n_agents = starts.shape[0]
    schedule = data['schedule']
    assert n_agents == len(schedule.keys()), "Plans for all agents"
    return data


def cost_ecbs(env, starts, goals):
    """get the average agent cost of this from ecbs
    returns: `float` and `-1` if planning was unsuccessful."""
    data = plan_ecbs(env, starts, goals)
    if data == INVALID:
        return data
    n_agents = starts.shape[0]
    schedule = data['schedule']
    cost_per_agent = []
    for i_a in range(n_agents):
        agent_key = 'agent'+str(i_a)
        assert agent_key in schedule.keys(), "Path for this agent"
        path = schedule[agent_key]
        cost_per_agent.append(path[-1]['t'])
    return sum(cost_per_agent) / n_agents


def blocks_ecbs(env, starts, goals) -> Tuple[int, int]:
    """Return numbers of vertex and edge blocks for this scenarios solution
    returns: (n_vertex_blocks, n_edge_blocks)"""
    data = plan_ecbs(env, starts, goals)
    if data == INVALID:
        return data
    blocks = data['blocks']
    n_agents = starts.shape[0]
    (n_vertex_blocks, n_edge_blocks) = (0, 0)
    for i_a in range(n_agents):
        agent_key = 'agent'+str(i_a)
        assert agent_key in blocks.keys(), "Blocks for this agent"
        blocks_pa = blocks[agent_key]
        if blocks_pa != 0:
            for contraint_type in blocks_pa.keys():
                if contraint_type == 'vertexConstraints':
                    n_vertex_blocks += len(blocks_pa[contraint_type])
                elif contraint_type == 'edgeConstraints':
                    n_edge_blocks += len(blocks_pa[contraint_type])
    return (n_vertex_blocks, n_edge_blocks)


@cachier(hash_params=tools.hasher)
def cost_independant(env, starts, goals):
    """what would be the average agent cost if the agents would be independent
    of each other"""
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


@cachier(hash_params=tools.hasher)
def cost_sim_decentralized_random(env, starts, goals):
    agents = to_agent_objects(env, starts, goals, Policy.RANDOM)
    (average_time, max_time, average_length, max_length, successful
     ) = run_a_scenario(
        env, agents, plot=False)
    if successful:
        return average_time + 1
    else:
        return INVALID
