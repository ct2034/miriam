import logging
from functools import lru_cache
from typing import Tuple

import networkx as nx
import numpy as np
import tools
from definitions import DEFAULT_TIMEOUT_S, FREE, INVALID
from networkx.algorithms import approximation
from planner.matteoantoniazzi_mapf.plan import (expanded_nodes_from_info,
                                                is_info_valid,
                                                sum_of_costs_from_info)
from sim.decentralized.agent import Agent
from sim.decentralized.policy import PolicyType
from sim.decentralized.runner import (is_environment_well_formed,
                                      to_agent_objects)

from scenarios.solvers import (SCHEDULE, cached_decentralized, cached_ecbs,
                               cached_icts, indep)

logging.getLogger('sim.decentralized.agent').setLevel(logging.ERROR)
logging.getLogger('sim.decentralized.runner').setLevel(logging.ERROR)
logging.getLogger(
    'planner.mapf_implementations').setLevel(logging.ERROR)

HIGHLEVELEXPANDED = 'highLevelExpanded'
LOWLEVELEXPANDED = 'lowLevelExpanded'
STATISTICS = 'statistics'


# static problem analysis ####################################################
def is_well_formed(env, starts, goals):
    """Check if the environment is well formed according to Cap2015"""
    agents = to_agent_objects(env, starts, goals)
    if agents is INVALID:
        return False
    return is_environment_well_formed(tuple(agents))


def cost_independent(env, starts, goals):
    """what would be the average agent cost if the agents would be independent
    of each other"""
    n_agents = starts.shape[0]
    paths = indep(env, starts, goals)
    if paths is INVALID:
        return INVALID
    return float(sum(map(lambda p: len(p)-1, paths))) / n_agents


def connectivity(env, starts, goals) -> float:
    """sum of inverse connectivities of start and goal pairs.
    Inverse is here `1/connectivity`, because we want to penalize low
    connectivity."""
    g = _gridmap_to_nx(env)
    cons = []
    for i_a in range(len(starts)):
        if all(starts[i_a] == goals[i_a]):
            cons.append(0)
        else:
            con = approximation.node_connectivity(
                g, tuple(starts[i_a]), tuple(goals[i_a]))
            cons.append(1 / con)
    return sum(cons)


def _eigenvector_centrality_dict(env):
    g = _gridmap_to_nx(env)
    ec = nx.eigenvector_centrality_numpy(g, max_iter=100)
    return ec


def uncentrality(env, starts, goals) -> float:
    """Hand crafted score that sums up eigenvector uncentralities for start and
    goal positions. Uncentrality is `1-centrality`"""
    # how much more to value uncentrality over number of agents
    SCALE_UNCENTRALITY: float = 3.
    ec = _eigenvector_centrality_dict(env)
    score = 0
    for i_a in range(len(starts)):
        score += 1 - ec[tuple(starts[i_a])] * SCALE_UNCENTRALITY
        score += 1 - ec[tuple(goals[i_a])] * SCALE_UNCENTRALITY
    return score


# ecbs ########################################################################
def cost_ecbs(env, starts, goals, timeout=DEFAULT_TIMEOUT_S, skip_cache=False):
    """get the average agent cost of this from ecbs
    returns: `float` and `-1` if planning was unsuccessful."""
    data = cached_ecbs(env, starts, goals, timeout=timeout,
                       skip_cache=skip_cache)
    if data == INVALID:
        return data
    try:
        n_agents = starts.shape[0]
        schedule = data[SCHEDULE]
        cost_per_agent = []
        for i_a in range(n_agents):
            agent_key = 'agent'+str(i_a)
            assert agent_key in schedule.keys(), "Path for this agent"
            path = schedule[agent_key]
            cost_per_agent.append(path[-1]['t'])
        return float(sum(cost_per_agent)) / n_agents
    except Exception as e:
        return INVALID


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


# decen #######################################################################
def cost_sim_decentralized_random(env, starts, goals, skip_cache=False):
    metrics = cached_decentralized(
        env, starts, goals, PolicyType.RANDOM, skip_cache)
    if metrics is INVALID:
        return INVALID
    (average_time, _, _, _, successful
     ) = metrics
    if successful:
        return average_time
    else:
        return INVALID


def cost_sim_decentralized_learned(env, starts, goals, skip_cache=False):
    metrics = cached_decentralized(
        env, starts, goals, PolicyType.LEARNED, skip_cache)
    if metrics is INVALID:
        return INVALID
    (average_time, _, _, _, successful
     ) = metrics
    if successful:
        return average_time
    else:
        return INVALID


# icts ########################################################################
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


# only dependant on graph #####################################################
def n_nodes(env):
    return np.count_nonzero(env == FREE)


def _gridmap_to_nx(env):
    a = Agent(env, np.array([0, 0]))
    return a.env_to_nx(env, None)


def n_edges(env):
    g = _gridmap_to_nx(env)
    return len(g.edges)


def mean_degree(env):
    g = _gridmap_to_nx(env)
    degrees = []
    for n in g.nodes:
        degrees.append(g.degree(n))
    return np.mean(degrees)


def tree_width(env):
    g = _gridmap_to_nx(env)
    w, _ = approximation.treewidth_min_degree(g)
    return w


def small_world_sigma(env):
    g = _gridmap_to_nx(env)
    return nx.sigma(g)


def small_world_omega(env):
    g = _gridmap_to_nx(env)
    return nx.omega(g)


def bridges(env):
    g = _gridmap_to_nx(env)
    return len(list(nx.bridges(g)))
