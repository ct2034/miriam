import logging
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
from definitions import POS, C
from sim.decentralized.agent import Agent

logging.basicConfig()
logger = logging.getLogger(__name__)

OBSERVATION_DISTANCE = 6


class IteratorType(Enum):
    WAITING = auto()
    BLOCKING1 = auto()
    BLOCKING3 = auto()
    EDGE_POLICY1 = auto()
    EDGE_POLICY3 = auto()


class SimIterationException(Exception):
    pass


def get_possible_next_agent_poses(
        agents: Tuple[Agent],
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


def get_poses_in_dt(agents: Tuple[Agent], dt: int) -> List[C]:
    """Get poses at time dt from now in the future, so `dt=0` is now."""
    poses: List[C] = []
    for a in agents:
        if a.is_at_goal(dt):
            assert a.goal is not None
            poses.append(a.goal)
        else:
            assert a.path is not None
            assert a.path_i is not None
            if a.has_gridmap:
                poses.append(a.path[a.path_i + dt][:-1])
            elif a.has_roadmap:
                poses.append(a.path[a.path_i + dt][0])
    return poses


def check_for_colissions(
        agents: Tuple[Agent],
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


def make_sure_agents_are_safe(agents: Tuple[Agent], ignore_fa: bool):
    """Assert that no too agents are in the same place"""
    poses = set()
    for a in agents:
        if not a.is_at_goal() or not ignore_fa:
            if a.has_gridmap:
                assert (tuple(a.pos) not in poses
                        ), "Two agents at the same place."
                poses.add(tuple(a.pos))
            elif a.has_roadmap:
                assert a.pos not in poses, "Two agents at the same place."
                poses.add(a.pos)


def has_at_least_one_agent_moved(
        agents: Tuple[Agent], agents_at_beginning: Tuple[Any, ...]
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


def iterate_waiting(agents: Tuple[Agent], ignore_finished_agents: bool) -> Tuple[List[int], List[float]]:
    """Given a set of agents, find possible next steps for each
    agent and move them there if possible."""
    # all that are not at their goals can generally procede. This is therefore
    # invariant of agents becoming finished in the iteration.
    can_proceed: List[bool] = list(map(lambda a: not a.is_at_goal(), agents))
    # assuming there are collisions
    there_are_collisions: bool = True

    time_slice = [0] * len(agents)
    space_slice = [0.] * len(agents)

    # how do agents look like at beginning?
    agents_at_beginning = tuple(map(lambda a: a.pos, agents))

    for i_a in range(len(agents)):
        # who is this agent seeing?
        for i_oa in [i for i in range(len(agents)) if i != i_a]:
            if np.linalg.norm(
                np.array(agents[i_a].pos) - np.array(agents[i_oa].pos)
            ) < OBSERVATION_DISTANCE:
                agents[i_a].policy.register_observation(
                    agents[i_oa].id,
                    agents[i_oa].path,
                    agents[i_oa].pos,
                    agents[i_oa].path_i
                )

    while(there_are_collisions):
        possible_next_agent_poses = get_possible_next_agent_poses(
            agents, can_proceed)

        # check collisions
        node_colissions, edge_colissions = check_for_colissions(
            agents, 0, possible_next_agent_poses, ignore_finished_agents)

        if (len(node_colissions.keys()) == 0 and
                len(edge_colissions.keys()) == 0):
            # nothing is blocked. everyone can continue
            there_are_collisions = False
        else:
            # we need to solve the blocks be not stepping some agents
            for pose, [i_a1, i_a2] in node_colissions.items():
                if (agents[i_a1].get_priority(agents[i_a2].id) >
                        agents[i_a2].get_priority(agents[i_a1].id)):
                    # a2 has lower prio
                    if not can_proceed[i_a2]:  # already blocked
                        can_proceed[i_a1] = False
                    can_proceed[i_a2] = False
                else:
                    # a1 has lower prio
                    if not can_proceed[i_a1]:  # already blocked
                        can_proceed[i_a2] = False
                    can_proceed[i_a1] = False
            for edge, [i_a1, i_a2] in edge_colissions.items():
                # edge collisions can not be solved by waiting. This is why we
                # block edges here. This makes this policy not strictly waiting.
                if (agents[i_a1].get_priority(agents[i_a2].id) >
                        agents[i_a2].get_priority(agents[i_a1].id)):
                    # a1 has higher prio
                    success2 = agents[i_a2].block_edge(
                        (edge[1], edge[0], 0))
                    if not success2:
                        success1 = agents[i_a1].block_edge(
                            (edge[0], edge[1], 0))
                        if not success1:
                            raise SimIterationException(
                                "Deadlock by edge collision")
                else:
                    # a2 has higher prio
                    success1 = agents[i_a1].block_edge(
                        (edge[0], edge[1], 0))
                    if not success1:
                        success2 = agents[i_a2].block_edge(
                            (edge[1], edge[0], 0))
                        if not success2:
                            raise SimIterationException(
                                "Deadlock by edge collision")

        if not any(can_proceed):
            # there is not one agent that can move
            raise SimIterationException("Deadlock by node collisions")

    for i_a, a in enumerate(agents):
        if can_proceed[i_a] and not a.is_at_goal():
            if a.has_gridmap:
                dx = 1.
            elif a.has_roadmap:
                pos_s = nx.get_node_attributes(a.env, POS)
                dx = float(torch.linalg.vector_norm(
                    torch.tensor(pos_s[a.pos]) -
                    torch.tensor(pos_s[possible_next_agent_poses[i_a]])
                ))
            a.make_next_step(possible_next_agent_poses[i_a])
            space_slice[i_a] = dx
        if not a.is_at_goal():
            time_slice[i_a] = 1

    make_sure_agents_are_safe(agents, ignore_finished_agents)
    assert has_at_least_one_agent_moved(
        agents, agents_at_beginning), "no agent has changed"

    return time_slice, space_slice


def iterate_blocking(agents: Tuple[Agent], lookahead: int, ignore_finished_agents: bool
                     ) -> Tuple[List[int], List[float]]:
    """Given a set of agents, find possible next steps for each
    agent and move them there if possible."""
    # get poses
    if agents[0].has_roadmap:
        pos_s = nx.get_node_attributes(agents[0].env, POS)

    # how do agents look like at beginning?
    poses_at_beginning = tuple(map(lambda a: a.pos, agents))

    for dt in range(lookahead - 1, -1, -1):
        # all that are not at their goals can generally procede. This is therefore
        # invariant of agents becoming finished in the iteration.
        can_proceed: List[bool] = list(
            map(lambda a: not a.is_at_goal(dt), agents))
        # assuming there are collisions
        there_are_collisions: bool = True

        poses_at_dt = get_poses_in_dt(agents, dt)
        for i_a in range(len(agents)):
            # who is this agent seeing?
            for i_oa in [i for i in range(len(agents)) if i != i_a]:
                if np.linalg.norm(
                    np.array(agents[i_a].pos) - np.array(agents[i_oa].pos)
                ) < OBSERVATION_DISTANCE:
                    agents[i_a].policy.register_observation(
                        agents[i_oa].id,
                        agents[i_oa].path,
                        poses_at_dt[i_oa],
                        agents[i_oa].get_path_i_not_none()
                    )  # observation regarding agent i_oa

        while(there_are_collisions and any(can_proceed) or
              there_are_collisions and dt == 0):

            # check collisions
            node_colissions, edge_colissions = check_for_colissions(
                agents, dt, ignore_finished_agents=ignore_finished_agents)

            if (len(node_colissions.keys()) == 0 and
                    len(edge_colissions.keys()) == 0):
                # nothing is blocked. everyone can continue
                there_are_collisions = False
            else:
                # we need to solve the blocks by blocking some agents
                for pose, [i_a1, i_a2] in node_colissions.items():
                    if agents[i_a1].has_roadmap:
                        pose = (pose,)
                    pose_to_block = pose + (dt+1,)
                    if (agents[i_a1].get_priority(agents[i_a2].id) >
                            agents[i_a2].get_priority(agents[i_a1].id)):
                        # a1 has higher prio
                        success2 = agents[i_a2].block_node(pose_to_block)
                        if not success2:
                            success1 = agents[i_a1].block_node(pose_to_block)
                            if not success1:
                                raise SimIterationException(
                                    "Deadlock by node collision")
                    else:
                        # a2 has higher prio
                        success1 = agents[i_a1].block_node(pose_to_block)
                        if not success1:
                            success2 = agents[i_a2].block_node(pose_to_block)
                            if not success2:
                                raise SimIterationException(
                                    "Deadlock by node collision")
                for edge, [i_a1, i_a2] in edge_colissions.items():
                    if (agents[i_a1].get_priority(agents[i_a2].id) >
                            agents[i_a2].get_priority(agents[i_a1].id)):
                        # a1 has higher prio
                        success2 = agents[i_a2].block_edge(
                            (edge[1], edge[0], dt))
                        if not success2:
                            success1 = agents[i_a1].block_edge(
                                (edge[0], edge[1], dt))
                            if not success1:
                                raise SimIterationException(
                                    "Deadlock by edge collision")
                    else:
                        # a2 has higher prio
                        success1 = agents[i_a1].block_edge(
                            (edge[0], edge[1], dt))
                        if not success1:
                            success2 = agents[i_a2].block_edge(
                                (edge[1], edge[0], dt))
                            if not success2:
                                raise SimIterationException(
                                    "Deadlock by edge collision")

    if not any(can_proceed):
        # there is not one agent that can move
        raise SimIterationException("Deadlock from unresolvable collision")

    time_slice: List[int] = [0] * len(agents)
    space_slice: List[float] = [0.] * len(agents)
    possible_next_poses = get_poses_in_dt(agents, 1)
    for i_a, a in enumerate(agents):
        if can_proceed[i_a] and not a.is_at_goal():
            if a.has_gridmap:
                dx = 1.
            elif a.has_roadmap:
                dx = float(np.linalg.norm(
                    pos_s[a.pos] -
                    pos_s[possible_next_poses[i_a]]
                ))
            space_slice[i_a] = dx
        if not a.is_at_goal():
            time_slice[i_a] = 1
        a.make_next_step(possible_next_poses[i_a])
        a.remove_all_blocks_and_replan()

    make_sure_agents_are_safe(agents, ignore_finished_agents)
    if not has_at_least_one_agent_moved(
            agents, poses_at_beginning):
        raise SimIterationException("Deadlock because of no progress")

    return time_slice, space_slice


def iterate_edge_policy(
    agents: Tuple[Agent],
    lookahead: int,
    ignore_finished_agents: bool
) -> Tuple[List[int], List[float]]:
    """An iterator that will ask a policy which edge to take in the even of a collision."""
    for a in agents:
        assert a.has_roadmap, "This function only works with roadmaps"

    solved = False
    RETRIES = 3
    i_try = 0

    all_colissions = []
    for dt in range(lookahead):
        all_colissions.append(check_for_colissions(
            agents, dt, None, ignore_finished_agents))
    agents_with_colissions = get_agents_in_col(all_colissions)
    logger.debug(f"all_colissions: {all_colissions}")

    # calling the policy for each agent that has colissions
    while (not solved) and i_try < RETRIES:
        logger.debug(f"agents with colissions: {agents_with_colissions}")
        next_nodes = []
        for i_a, a in enumerate(agents):
            if i_a in agents_with_colissions:
                assert hasattr(a.policy, "get_edge"), \
                    "Needs edge-based policy"
                next_nodes.append(a.policy.get_edge(agents))  # type: ignore
            else:
                next_nodes.append(a.what_is_next_step())
        next_collisions = check_for_colissions(
            agents, 0, next_nodes, ignore_finished_agents)
        solved = not any(next_collisions)
        logger.debug(
            f"try {i_try}, solved: {solved}, next_collisions: {next_collisions}")
        new_agents_with_colissions = get_agents_in_col([next_collisions])
        agents_with_colissions.update(new_agents_with_colissions)
        i_try += 1
    if i_try == RETRIES:
        raise SimIterationException(f"Failed to solve after {RETRIES} tries")
    else:
        for i_a, a in enumerate(agents):
            a.make_next_step(next_nodes[i_a])
            a.remove_all_blocks_and_replan()

    time_slice: List[int] = [0] * len(agents)
    space_slice: List[float] = [0.] * len(agents)
    return time_slice, space_slice


def get_agents_in_col(all_colissions):
    agents_with_colissions = set()
    for i_t in range(len(all_colissions)):
        nodecol, edgecol = all_colissions[i_t]
        for node, [i_a1, i_a2] in nodecol.items():
            agents_with_colissions.update([i_a1, i_a2])
        for edge, [i_a1, i_a2] in edgecol.items():
            agents_with_colissions.update([i_a1, i_a2])
    return agents_with_colissions


def get_iterator_fun(type: IteratorType):
    if type is IteratorType.WAITING:
        return iterate_waiting
    elif type is IteratorType.BLOCKING1:
        return lambda agents, ignore_fa: iterate_blocking(agents, 1, ignore_fa)
    elif type is IteratorType.BLOCKING3:
        return lambda agents, ignore_fa: iterate_blocking(agents, 3, ignore_fa)
    elif type is IteratorType.EDGE_POLICY1:
        return lambda agents, ignore_fa: iterate_edge_policy(agents, 1, ignore_fa)
    elif type is IteratorType.EDGE_POLICY3:
        return lambda agents, ignore_fa: iterate_edge_policy(agents, 3, ignore_fa)
