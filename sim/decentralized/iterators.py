from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sim.decentralized.agent import Agent

OBSERVATION_DISTANCE = 6


class IteratorType(Enum):
    WAITING = auto()
    BLOCKING1 = auto()
    BLOCKING3 = auto()


class SimIterationException(Exception):
    pass


def get_possible_next_agent_poses(
        agents: Tuple[Agent],
        can_proceed: List[bool]) -> np.ndarray:
    """Where would the agents be if they would be allowed to move to the next
    step in their paths if they have a true in `can_proceed`."""
    possible_next_agent_poses = np.zeros((len(agents), 2), dtype=int)
    # prepare step
    for i_a in range(len(agents)):
        if can_proceed[i_a]:
            possible_next_agent_poses[i_a, :] = agents[i_a].what_is_next_step()
        else:
            possible_next_agent_poses[i_a, :] = agents[i_a].pos
    return possible_next_agent_poses


def get_poses_in_dt(agents: Tuple[Agent], dt: int) -> np.ndarray:
    """Get poses at time dt from now in the future, so `dt=0` is now."""
    poses: np.ndarray = np.zeros((len(agents), 2), dtype=int)
    for i_a, a in enumerate(agents):
        assert a.path is not None
        assert a.path_i is not None
        if a.is_at_goal(dt):
            poses[i_a] = a.goal
        else:
            poses[i_a] = a.path[a.path_i + dt, :2]
    return poses


def check_for_colissions(
        agents: Tuple[Agent],
        dt: int = 0,
        possible_next_agent_poses: Optional[np.ndarray] = None) -> Tuple[Dict[Any, Any], Dict[Any, Any]]:
    """check for two agents going to meet at one vertex or two agents using
    the same edge."""
    node_colissions = {}
    edge_colissions = {}
    if possible_next_agent_poses is not None:
        next_poses = possible_next_agent_poses
    else:
        next_poses = get_poses_in_dt(agents, 1 + dt)
    current_poses = get_poses_in_dt(agents, dt)
    # ignoring finished agents
    for i_a in range(len(agents)):
        if not agents[i_a].is_at_goal(dt):
            for i_oa in [i for i in range(len(agents)) if i > i_a]:
                if not agents[i_oa].is_at_goal(dt):
                    if (next_poses[i_a] ==
                            next_poses[i_oa]).all():
                        node_colissions[tuple(next_poses[i_a])] = [i_a, i_oa]
                    if ((next_poses[i_a] == current_poses[i_oa]).all() and
                            (next_poses[i_oa] == current_poses[i_a]).all()):
                        edge = [tuple(current_poses[i_a]),
                                tuple(next_poses[i_a])]
                        edge_colissions[tuple(edge)] = [i_a, i_oa]
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


def make_sure_agents_are_safe(agents: Tuple[Agent]):
    """Assert that no too agents are in the same place"""
    poses = set()
    for a in agents:
        if not a.is_at_goal():
            assert tuple(a.pos) not in poses, "Two agents at the same place."
            poses.add(tuple(a.pos))


def has_at_least_one_agent_moved(
        agents: Tuple[Agent], agents_at_beginning: Tuple[Tuple[Any, ...], ...]
) -> bool:
    """given the set of agents from the start, have they changed now?"""
    for i_a in range(len(agents)):
        if any(agents[i_a].pos != agents_at_beginning[i_a]):
            return True
    return False


def iterate_waiting(agents: Tuple[Agent]) -> Tuple[List[int], List[int]]:
    """Given a set of agents, find possible next steps for each
    agent and move them there if possible."""
    # all that are not at their goals can generally procede. This is therefore
    # invariant of agents becoming finished in the iteration.
    can_proceed: List[bool] = list(map(lambda a: not a.is_at_goal(), agents))
    # assuming there are collisions
    there_are_collisions: bool = True

    time_slice = [0] * len(agents)
    space_slice = [0] * len(agents)

    # how do agents look like at beginning?
    agents_at_beginning = tuple(map(lambda a: a.pos, agents))

    for i_a in range(len(agents)):
        # who is this agent seeing?
        for i_oa in [i for i in range(len(agents)) if i != i_a]:
            if np.linalg.norm(
                agents[i_a].pos - agents[i_oa].pos
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
            agents, 0, possible_next_agent_poses)

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

    for i_a in range(len(agents)):
        if can_proceed[i_a] and not agents[i_a].is_at_goal():
            agents[i_a].make_next_step(possible_next_agent_poses[i_a, :])
            space_slice[i_a] = 1
        if not agents[i_a].is_at_goal():
            time_slice[i_a] = 1

    make_sure_agents_are_safe(agents)
    assert has_at_least_one_agent_moved(
        agents, agents_at_beginning), "no agent has changed"

    return time_slice, space_slice


def iterate_blocking(agents: Tuple[Agent], lookahead: int
                     ) -> Tuple[List[int], List[int]]:
    """Given a set of agents, find possible next steps for each
    agent and move them there if possible."""
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
                    agents[i_a].pos - agents[i_oa].pos
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
                agents, dt)

            if (len(node_colissions.keys()) == 0 and
                    len(edge_colissions.keys()) == 0):
                # nothing is blocked. everyone can continue
                there_are_collisions = False
            else:
                # we need to solve the blocks by blocking some agents
                for pose, [i_a1, i_a2] in node_colissions.items():
                    pose_to_block = (pose[0], pose[1], dt+1)
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

    possible_next_agent_poses = get_poses_in_dt(agents, 1)
    time_slice: List[int] = [0] * len(agents)
    space_slice: List[int] = [0] * len(agents)
    for i_a in range(len(agents)):
        if can_proceed[i_a] and not agents[i_a].is_at_goal():
            agents[i_a].make_next_step(possible_next_agent_poses[i_a, :])
            space_slice[i_a] = 1
        if not agents[i_a].is_at_goal():
            time_slice[i_a] = 1
        agents[i_a].remove_all_blocks_and_replan()

    make_sure_agents_are_safe(agents)
    assert has_at_least_one_agent_moved(
        agents, poses_at_beginning), "no agent has changed"

    return time_slice, space_slice


# -> ((Tuple[Agent]) -> Tuple[List[int], List[int]]):
def get_iterator_fun(type: IteratorType):
    if type is IteratorType.WAITING:
        return iterate_waiting
    elif type is IteratorType.BLOCKING1:
        return lambda x: iterate_blocking(x, 1)
    elif type is IteratorType.BLOCKING3:
        return lambda x: iterate_blocking(x, 3)
