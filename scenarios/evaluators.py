import logging
from functools import lru_cache
from os import EX_OSFILE
from typing import Tuple

import numpy as np
import tools
from definitions import DEFAULT_TIMEOUT_S, FREE, INVALID
from planner.matteoantoniazzi_mapf.plan import (expanded_nodes_from_info,
                                                is_info_valid,
                                                sum_of_costs_from_info)
from sim.decentralized.agent import Agent
from sim.decentralized.runner import is_environment_well_formed, run_a_scenario

from scenarios import storage
from scenarios.solvers import SCHEDULE, ecbs, icts, indep, to_agent_objects

logging.getLogger('sim.decentralized.agent').setLevel(logging.ERROR)
logging.getLogger('sim.decentralized.runner').setLevel(logging.ERROR)
logging.getLogger(
    'planner.policylearn.libMultiRobotPlanning').setLevel(logging.ERROR)

HIGHLEVELEXPANDED = 'highLevelExpanded'
LOWLEVELEXPANDED = 'lowLevelExpanded'
STATISTICS = 'statistics'


def is_well_formed(env, starts, goals):
    """Check if the environment is well formed according to Cap2015"""
    agents = to_agent_objects(env, starts, goals)
    if agents is INVALID:
        return False
    return is_environment_well_formed(tuple(agents))


def cached_ecbs(env, starts, goals,
                timeout=DEFAULT_TIMEOUT_S, skip_cache=False):
    if skip_cache:
        data = ecbs(env, starts, goals, timeout=timeout)
    else:
        scenario = (env, starts, goals)
        if storage.has_result(scenario, storage.ResultType.ECBS_DATA):
            data = storage.get_result(
                scenario, storage.ResultType.ECBS_DATA)
        else:
            data = ecbs(env, starts, goals, timeout=timeout)
            storage.save_result(scenario, storage.ResultType.ECBS_DATA, data)
    return data


def cost_ecbs(env, starts, goals, timeout=DEFAULT_TIMEOUT_S, skip_cache=False):
    """get the average agent cost of this from ecbs
    returns: `float` and `-1` if planning was unsuccessful."""
    data = cached_ecbs(env, starts, goals, timeout=timeout,
                       skip_cache=skip_cache)
    if data == INVALID:
        return data
    n_agents = starts.shape[0]
    schedule = data[SCHEDULE]
    cost_per_agent = []
    for i_a in range(n_agents):
        agent_key = 'agent'+str(i_a)
        assert agent_key in schedule.keys(), "Path for this agent"
        path = schedule[agent_key]
        cost_per_agent.append(path[-1]['t'])
    return float(sum(cost_per_agent)) / n_agents


def expanded_nodes_ecbs(env, starts, goals, timeout=DEFAULT_TIMEOUT_S):
    data = cached_ecbs(env, starts, goals, timeout=timeout)
    if data == INVALID:
        return data
    return data[STATISTICS][HIGHLEVELEXPANDED]


def blocks_ecbs(env, starts, goals) -> Tuple[int, int]:
    """Return numbers of vertex and edge blocks for this scenarios solution
    returns: (n_vertex_blocks, n_edge_blocks)"""
    data = cached_ecbs(env, starts, goals)
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


def cost_independent(env, starts, goals):
    """what would be the average agent cost if the agents would be independent
    of each other"""
    n_agents = starts.shape[0]
    paths = indep(env, starts, goals)
    if paths is INVALID:
        return INVALID
    return float(sum(map(lambda p: len(p)-1, paths))) / n_agents


def cost_sim_decentralized_random(env, starts, goals):
    agents = to_agent_objects(env, starts, goals)
    if agents is INVALID:
        return INVALID
    (average_time, _, _, _, successful
     ) = run_a_scenario(
        env, agents, plot=False)
    if successful:
        return average_time + 1
    else:
        return INVALID


def cached_icts(env, starts, goals,
                timeout=DEFAULT_TIMEOUT_S, skip_cache=False):
    scenario = (env, starts, goals)
    if skip_cache:
        info = icts(env, starts, goals, timeout)
    else:
        if storage.has_result(scenario, storage.ResultType.ICTS_INFO):
            info = storage.get_result(
                scenario, storage.ResultType.ICTS_INFO)
        else:
            info = icts(env, starts, goals, timeout)
            storage.save_result(scenario, storage.ResultType.ICTS_INFO, info)
    return info


def expanded_nodes_icts(env, starts, goals, timeout=DEFAULT_TIMEOUT_S):
    info = cached_icts(env, starts, goals, timeout=timeout)
    if is_info_valid(info):
        return expanded_nodes_from_info(info)
    else:
        return INVALID


def cost_icts(env, starts, goals, timeout=DEFAULT_TIMEOUT_S, skip_cache=False):
    n_agents = starts.shape[0]
    info = cached_icts(env, starts, goals, timeout=timeout,
                       skip_cache=skip_cache)
    if is_info_valid(info):
        return float(sum_of_costs_from_info(info)) / n_agents
    else:
        return INVALID


def n_nodes(env):
    return np.count_nonzero(env == FREE)


def _gridmap_to_nx(env):
    a = Agent(env, [0, 0])
    return a.gridmap_to_nx(env, None)


def n_edges(env):
    g = _gridmap_to_nx(env)
    return len(g.edges)


def mean_degree(env):
    g = _gridmap_to_nx(env)
    degrees = []
    for n in g.nodes:
        degrees.append(g.degree(n))
    return np.mean(degrees)
