import logging
from enum import Enum, auto
from multiprocessing import Pool
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import torch
from definitions import POS, C
from planner.mapf_implementations.libMultiRobotPlanning.tools.annotate_roadmap import \
    check_edges
from planner.mapf_implementations.libMultiRobotPlanning.tools.collision import \
    precheck_bounding_box
from sim.decentralized.agent import Agent

logging.basicConfig()
logger = logging.getLogger(__name__)

OBSERVATION_DISTANCE = 6


class IteratorType(Enum):
    LOOKAHEAD1 = auto()
    LOOKAHEAD2 = auto()
    LOOKAHEAD3 = auto()


class SimIterationException(Exception):
    pass


def get_possible_next_agent_poses(
        agents: Tuple[Agent, ...],
        can_proceed: List[bool]) -> List[C]:
    """Where would the agents be if they would be allowed to move to the next
    step in their paths if they have a true in `can_proceed`."""
    possible_next_agent_poses: List[C] = []
    # prepare step
    for i_a in range(len(agents)):
        if can_proceed[i_a]:
            possible_next_agent_poses.append(agents[i_a].what_is_next_step())
        else:
            possible_next_agent_poses.append(agents[i_a].pos)
    return possible_next_agent_poses


def get_poses_in_dt(agents: Tuple[Agent, ...], dt: int) -> List[C]:
    """Get poses at time dt from now in the future, so `dt=0` is now."""
    poses: List[C] = []
    for a in agents:
        if a.is_at_goal(dt):
            assert a.goal is not None
            poses.append(a.goal)
        else:
            assert a.path is not None
            assert a.path_i is not None
            poses.append(a.path[a.path_i + dt])
    return poses


def check_for_colissions(
        agents: Tuple[Agent, ...],
        dt: int = 0,
        possible_next_agent_poses: Optional[List[C]] = None,
        ignore_finished_agents: bool = True) -> Tuple[Dict[Any, Any], Dict[Any, Any]]:
    """check for two agents going to meet at one vertex or two agents using
    the same edge."""
    node_colissions: Dict[C, List[int]] = {}
    edge_colissions = {}
    if possible_next_agent_poses is not None:
        next_poses: List[C] = possible_next_agent_poses
    else:
        next_poses = get_poses_in_dt(agents, 1 + dt)
    current_poses = get_poses_in_dt(agents, dt)
    # ignoring finished agents
    for i_a in range(len(agents)):
        if not agents[i_a].is_at_goal(dt) or not ignore_finished_agents:
            for i_oa in [i for i in range(len(agents)) if i > i_a]:
                if not agents[i_oa].is_at_goal(dt) or not ignore_finished_agents:
                    if next_poses[i_a] == next_poses[i_oa]:
                        node_colissions[next_poses[i_a]] = [i_a, i_oa]
                    if ((next_poses[i_a] == current_poses[i_oa]) and
                            (next_poses[i_oa] == current_poses[i_a])):
                        edge = (current_poses[i_a],
                                next_poses[i_a])
                        edge_colissions[edge] = [i_a, i_oa]
                        #####################################################
                        #                                                   #
                        #  t     i_a   i_oa                                 #
                        #  |        \ /                                     #
                        #  |         X            \    edge_colissions[     #
                        #  |        / \        ----\     ((0, 0), (1, 0))   #
                        #  |       |/ \|       ----/   ] = [i_a, i_oa]      #
                        #  v    i_oa   i_a        /                         #
                        #                                                   #
                        # node (0, 0) (1, 0)                                #
                        #                                                   #
                        #####################################################
    return node_colissions, edge_colissions


def make_sure_agents_are_safe(agents: Tuple[Agent, ...], ignore_fa: bool):
    """Assert that no too agents are in the same place"""
    poses = set()
    for a in agents:
        if not a.is_at_goal() or not ignore_fa:
            assert a.pos not in poses, "Two agents at the same place."
            poses.add(a.pos)


def has_at_least_one_agent_moved(
        agents: Tuple[Agent, ...], agents_at_beginning: Tuple[Any, ...]
) -> bool:
    """given the set of agents from the start, have they changed now?"""
    for i_a in range(len(agents)):
        if agents[i_a].has_gridmap:
            if agents[i_a].pos != agents_at_beginning[i_a]:
                return True
        elif agents[i_a].has_roadmap:
            if agents[i_a].pos != agents_at_beginning[i_a]:
                return True
    return False


def check_motion_col(g: nx.Graph, radius: float,
                     node_s_start: List[C], node_s_end: List[C]) -> Set[int]:
    pos_s = nx.get_node_attributes(g, POS)
    n_agents = len(node_s_start)
    assert len(node_s_end) == n_agents,\
        "node_s_start and node_s_end must have same length"
    E = np.diag([radius, radius])
    colliding_agents = set()
    edges_to_check: List[Tuple[
        int, int, np.ndarray,  # i, j, E
        np.ndarray, np.ndarray, np.ndarray, np.ndarray  # p0, p1, q0, q1
    ]] = []
    for i_a1 in range(n_agents):
        for i_a2 in range(i_a1 + 1, n_agents):
            p0 = np.array(pos_s[node_s_start[i_a1]])
            p1 = np.array(pos_s[node_s_end[i_a1]])
            q0 = np.array(pos_s[node_s_start[i_a2]])
            q1 = np.array(pos_s[node_s_end[i_a2]])
            if precheck_bounding_box(E, p0, p1, q0, q1):
                edges_to_check.append((i_a1, i_a2, E, p0, p1, q0, q1))
    results = [check_edges(*edge[2:]) for edge in edges_to_check]
    for result, (i_a1, i_a2, _, _, _, _, _) in zip(results, edges_to_check):
        if result:
            colliding_agents.add(i_a1)
            colliding_agents.add(i_a2)
    return colliding_agents


def iterate_edge_policy(
    agents: Tuple[Agent, ...],
    lookahead: int,
    ignore_finished_agents: bool
) -> Tuple[List[int], List[float]]:
    """An iterator that will ask a policy which edge to take in the even of a collision."""
    assert agents[0].radius is not None, "radius must be set"
    assert all(a.radius == agents[0].radius for a in agents),\
        "all radii must be equal"

    RETRIES = 3
    i_try = 0
    space_slice: List[float] = [0.] * len(agents)
    time_slice: List[int] = [1] * len(agents)
    pos = nx.get_node_attributes(agents[0].env, POS)

    poses_at_beginning = list(map(lambda a: a.pos, agents))
    all_colissions = []
    for dt in range(lookahead):
        all_colissions.append(check_for_colissions(
            agents, dt, None, ignore_finished_agents))
    agents_with_colissions = get_agents_in_col(all_colissions)
    logger.debug(f"all_colissions: {all_colissions}")

    # calling the policy for each agent that has colissions
    next_nodes: List[C] = [-1] * len(agents)
    solved = False
    while (not solved) and i_try < RETRIES:
        logger.debug(f"agents with colissions: {agents_with_colissions}")
        for i_a, a in enumerate(agents):
            if i_a in agents_with_colissions:
                assert hasattr(a.policy, "get_edge"), \
                    "Needs edge-based policy"
                next_nodes[i_a] = a.policy.get_edge(  # type: ignore
                    agents, agents_with_colissions)  # type: ignore
            else:
                next_nodes[i_a] = a.what_is_next_step()
        next_collisions = check_for_colissions(
            agents, 0, next_nodes, ignore_finished_agents)
        new_agents_with_colissions = get_agents_in_col([next_collisions])
        new_agents_with_colissions.update(check_motion_col(
            agents[0].env, agents[0].radius, next_nodes, poses_at_beginning))
        solved = not any(new_agents_with_colissions)
        logger.debug(
            f"try {i_try}, solved: {solved}, next_collisions: {next_collisions}")
        agents_with_colissions.update(new_agents_with_colissions)
        i_try += 1
    if i_try == RETRIES:
        raise SimIterationException(f"Failed to solve after {RETRIES} tries")
    else:
        for i_a, a in enumerate(agents):
            if not a.is_at_goal():
                time_slice[i_a] = 1
            space_slice[i_a] = float(np.linalg.norm(
                np.array(pos[a.pos], dtype=np.float32) -
                np.array(pos[next_nodes[i_a]], dtype=np.float32)
            ))
            a.make_this_step(next_nodes[i_a])
            a.replan()
            a.policy.step()  # type: ignore

    return time_slice, space_slice


def get_agents_in_col(all_colissions) -> Set[int]:
    agents_with_colissions = set()
    for i_t in range(len(all_colissions)):
        nodecol, edgecol = all_colissions[i_t]
        for node, [i_a1, i_a2] in nodecol.items():
            agents_with_colissions.update([i_a1, i_a2])
        for edge, [i_a1, i_a2] in edgecol.items():
            agents_with_colissions.update([i_a1, i_a2])
    return agents_with_colissions


def get_iterator_fun(type: IteratorType):
    if type is IteratorType.LOOKAHEAD1:
        return lambda agents, ignore_fa: iterate_edge_policy(agents, 1, ignore_fa)
    elif type is IteratorType.LOOKAHEAD2:
        return lambda agents, ignore_fa: iterate_edge_policy(agents, 2, ignore_fa)
    elif type is IteratorType.LOOKAHEAD3:
        return lambda agents, ignore_fa: iterate_edge_policy(agents, 3, ignore_fa)
