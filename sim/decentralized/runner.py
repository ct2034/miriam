#!/usr/bin/env python3

import itertools
import logging
import random
from functools import lru_cache
from typing import *

import networkx as nx
import numpy as np
import timeout_decorator
from matplotlib import cm
from matplotlib import pyplot as plt
from sim.decentralized.agent import Agent
from sim.decentralized.policy import PolicyType
from tools import ProgressBar

logging.basicConfig()
logger = logging.getLogger(__name__)


class SimIterationException(Exception):
    pass


def initialize_environment(size: int, fill: float, seed: Any = random.random()):
    """Make a square map with edge length `size` and `fill` (0..1) obstacle
    ratio.

    :param size: side length of the square (in pixels)
    :type size: int
    :param fill: percentage of map to fill
    :type fill: float
    :return: the environment
    :rtype: np.ndarray
    """
    random.seed(seed)
    environent = np.zeros([size, size], dtype=np.int8)
    n_to_fill = int(fill * size ** 2)
    to_fill = random.sample(
        list(itertools.product(range(size), repeat=2)), k=n_to_fill)
    for cell in to_fill:
        environent[cell] = 1
    return environent


def initialize_new_agent(
        env: np.ndarray, agents: List[Agent], policy: PolicyType,
        tight_placement: bool = False) -> Agent:
    """Place new agent in the environment, where no obstacle or other agent
    is.

    :param tight_placement: if false, start and goal places are sampled from
    the same distribution
    :raises AssertionError if no space is left
    :return: the agent
    """
    if tight_placement:  # starts can be at other goals
        env_with_agents = env.copy()
        env_with_goals = env.copy()
        for a in agents:
            env_with_agents[tuple(a.pos)] = 1
            assert a.goal is not None, "Agent should have a goal"
            env_with_goals[tuple(a.goal)] = 1
        no_obstacle_nor_agent = np.where(env_with_agents == 0)
        no_obstacle_nor_goal = np.where(env_with_goals == 0)
        assert len(no_obstacle_nor_agent[0]
                   ) > 0, "Possible poses should be left"
        assert len(no_obstacle_nor_goal[0]
                   ) > 0, "Possible poses should be left"
        pos = random.choice(np.transpose(no_obstacle_nor_agent))
        a = Agent(env, pos, policy)  # we have the agent

        # now finding the goal
        goal = random.choice(np.transpose(no_obstacle_nor_goal))
        a.give_a_goal(goal)
    else:  # no tight placement: sample starts and goals from same distribution
        env_with_agents_and_goals = env.copy()
        for a in agents:
            env_with_agents_and_goals[tuple(a.pos)] = 1
            assert a.goal is not None, "Agent should have a goal"
            env_with_agents_and_goals[tuple(a.goal)] = 1
        no_obstacle_nor_agent_or_goal = np.where(
            env_with_agents_and_goals == 0)
        assert len(no_obstacle_nor_agent_or_goal[0]
                   ) > 0, "Possible poses should be left"
        pos = random.choice(np.transpose(no_obstacle_nor_agent_or_goal))
        a = Agent(env, pos, policy)  # we have the agent

        # now finding the goal
        env_with_agents_and_goals[tuple(pos)] = 1
        no_obstacle_nor_agent_or_goal = np.where(
            env_with_agents_and_goals == 0)
        assert len(no_obstacle_nor_agent_or_goal[0]
                   ) > 0, "Possible poses should be left"
        goal = random.choice(np.transpose(no_obstacle_nor_agent_or_goal))
        a.give_a_goal(goal)

    return a


def initialize_agents(
        env: np.ndarray, n_agents: int, policy: PolicyType,
        tight_placement: bool = False, seed: Any = random.random()
) -> Tuple[Agent, ...]:
    """Initialize `n_agents` many agents in unique, free spaces of
    `environment`, (not colliding with each other)."""
    random.seed(seed)
    agents: List[Agent] = []  # starting with a list for easy inserting
    for _ in range(n_agents):
        agent = initialize_new_agent(
            env, agents, policy, tight_placement)
        agents.append(agent)
    return tuple(agents)  # returning tuple because it can be immutable now


@lru_cache(maxsize=512)
def is_environment_well_formed(agents: Tuple[Agent]) -> bool:
    """Check if the environment is well formed according to Cap2015"""
    for a in agents:
        blocks: List[Tuple[Any, ...]] = []
        for other_a in [ia for ia in agents if ia != a]:
            blocks.append(tuple(other_a.pos))
            assert other_a.goal is not None, "Other agent should have a goal"
            blocks.append(tuple(other_a.goal))
        if not a.is_there_path_with_node_blocks(blocks):
            return False
    return True


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


def check_for_colissions(
        agents: Tuple[Agent],
        nxt_poses: np.ndarray) -> Tuple[Dict[Any, Any], Dict[Any, Any]]:
    """check for two agents going to meet at one vertex or two agents using
    the same edge."""
    node_colissions = {}
    edge_colissions = {}
    # ignoring finished agents
    for i_a in range(len(agents)):
        if not agents[i_a].is_at_goal():
            for i_oa in [i for i in range(len(agents)) if i > i_a]:
                if not agents[i_oa].is_at_goal():
                    if (nxt_poses[i_a] ==
                            nxt_poses[i_oa]).all():
                        node_colissions[tuple(nxt_poses[i_a])] = [i_a, i_oa]
                    if ((nxt_poses[i_a] == agents[i_oa].pos).all() and
                            (nxt_poses[i_oa] == agents[i_a].pos).all()):
                        edge = [tuple(agents[i_a].pos),
                                tuple(nxt_poses[i_a])]
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


# @timeout_decorator.timeout(1)
def iterate_sim(agents: Tuple[Agent]) -> Tuple[List[int], List[int]]:
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

    while(there_are_collisions):
        possible_next_agent_poses = get_possible_next_agent_poses(
            agents, can_proceed)

        # check collisions
        node_colissions, edge_colissions = check_for_colissions(
            agents, possible_next_agent_poses)

        if (len(node_colissions.keys()) == 0 and
                len(edge_colissions.keys()) == 0):
            # nothing is blocked. everyone can continue
            there_are_collisions = False
        else:
            # we need to solve the blocks be not stepping some agents
            for pose, [i_a1, i_a2] in node_colissions.items():
                if not can_proceed[i_a1]:  # already blocked
                    can_proceed[i_a1] = True
                    can_proceed[i_a2] = False  # block other agent
                elif not can_proceed[i_a2]:  # already blocked
                    can_proceed[i_a1] = True
                    can_proceed[i_a1] = False  # block other agent
                elif agents[i_a1].get_priority() > agents[i_a2].get_priority():
                    can_proceed[i_a2] = False  # has lower prio
                else:
                    can_proceed[i_a1] = False  # has lower prio
            for edge, [i_a1, i_a2] in edge_colissions.items():
                if agents[i_a1].get_priority() > agents[i_a2].get_priority():
                    # a1 has higher prio
                    success = agents[i_a2].block_edge(edge[1], edge[0])
                    if not success:
                        success = agents[i_a1].block_edge(edge[0], edge[1])
                        if not success:
                            raise SimIterationException(
                                "Deadlock by edge collision")
                else:
                    # a2 has higher prio
                    success = agents[i_a1].block_edge(edge[0], edge[1])
                    if not success:
                        success = agents[i_a2].block_edge(edge[1], edge[0])
                        if not success:
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


def are_all_agents_at_their_goals(agents: List[Agent]) -> bool:
    """Returns true iff all agents are at their respective goals."""
    return all(map(lambda a: a.is_at_goal(), agents))


def check_time_evaluation(time_progress, space_progress):
    average_time = sum(time_progress) / len(time_progress)
    max_time = max(time_progress)
    average_length = sum(space_progress) / len(space_progress)
    max_length = max(space_progress)
    return average_time, max_time, average_length, max_length


def sample_and_run_a_scenario(size, n_agents, policy, plot, seed):
    env = initialize_environment(size, .2, seed)
    agents = initialize_agents(env, n_agents, policy, seed)
    return run_a_scenario(env, agents, plot)


def run_a_scenario(env, agents, plot):
    n_agents = len(agents)
    # evaluation parameters
    time_progress = np.zeros([n_agents])
    space_progress = np.zeros([n_agents])
    # iterate
    successful = 0
    try:
        while not are_all_agents_at_their_goals(agents):
            if plot:  # pragma: no cover
                plot_env_agents(env, agents)
            time_slice, space_slice = iterate_sim(agents)
            time_progress += time_slice
            space_progress += space_slice
        successful = 1
    except SimIterationException as e:  # pragma: no cover
        logger.warning(e)

    return check_time_evaluation(
        time_progress,
        space_progress) + (successful, )


def plot_env_agents(environent: np.ndarray,
                    agents: np.ndarray):  # pragma: no cover
    """Plot the environment map with `x` coordinates to the right, `y` up.
    Occupied by colorful agents and their paths."""
    # map
    image = environent * -.5 + 1
    image = np.swapaxes(image, 0, 1)
    fig, ax = plt.subplots()
    c = ax.imshow(image, cmap='gray', vmin=0, vmax=1)
    ax.set_aspect('equal')
    baserange = np.arange(environent.shape[0], step=2)
    ax.set_xticks(baserange)
    ax.set_xticklabels(map(str, baserange))
    ax.set_yticks(baserange)
    ax.set_yticklabels(map(str, baserange))
    colormap = cm.tab10.colors

    # gridlines
    for i_x in range(environent.shape[0]-1):
        ax.plot(
            [i_x + .5] * 2,
            [-.5, environent.shape[1]-.5],
            color='dimgrey',
            linewidth=.5
        )
    for i_y in range(environent.shape[1]-1):
        ax.plot(
            [-.5, environent.shape[0]-.5],
            [i_y + .5] * 2,
            color='dimgrey',
            linewidth=.5
        )

    # agents
    for i_a, a in enumerate(agents):
        ax.plot(a.pos[0], a.pos[1], markersize=10,
                marker='o', color=colormap[i_a])
        ax.plot(a.goal[0], a.goal[1], markersize=10,
                marker='x', color=colormap[i_a])
        ax.plot(a.path[:, 0], a.path[:, 1], color=colormap[i_a])

    plt.show()


def plot_evaluations(evaluations: Dict[PolicyType, np.ndarray],
                     evaluation_names: List[str]):  # pragma: no cover
    n_policies = len(evaluations.keys())
    colormap = cm.tab10.colors
    data_shape = (n_policies, ) + list(evaluations.values())[0].shape
    data = np.empty(data_shape)

    policy_names = []
    subplot_basenr = 100 + 10 * len(evaluation_names) + 1
    for i_p, policy in enumerate(evaluations.keys()):
        data[i_p, :, :] = evaluations[policy]
        policy_names.append(str(policy).replace('Policy.', ''))

    plt.figure(figsize=[16, 9])

    for i_e, evaluation_name in enumerate(evaluation_names):
        ax = plt.subplot(subplot_basenr + i_e)
        parts = ax.violinplot(np.transpose(data[:, i_e, :]), showmeans=True)
        for i_pc, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colormap[i_pc])
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
            vp = parts[partname]
            vp.set_edgecolor('k')
        ax.set_title(evaluation_name)
        plt.xticks(range(1, n_policies + 1), policy_names)

    plt.show()


def evaluate_policies(size=10, n_agents=10, runs=100, plot_eval=True):
    """run the simulation with all policies"""
    evaluations = {}
    evaluation_names = [
        "average_time",
        "max_time",
        "average_length",
        "max_length",
        "successful"
    ]
    pb = ProgressBar("evaluate_policies", len(PolicyType)*runs, 5)

    for policy in PolicyType:
        logger.info(f"policy: {policy.name}")
        evaluation_per_policy = np.empty([len(evaluation_names), 0])
        for i_r in range(runs):
            print(i_r)
            random.seed(i_r)
            results = sample_and_run_a_scenario(
                size, n_agents, policy, False, i_r)
            evaluation_per_policy = np.append(
                evaluation_per_policy, np.transpose([results]), axis=1)
            pb.progress()
        evaluations[policy] = evaluation_per_policy
    pb.end()

    if plot_eval:  # pragma: no cover
        plot_evaluations(evaluations, evaluation_names)
    return (evaluations, evaluation_names)


if __name__ == "__main__":  # pragma: no cover
    evaluate_policies(32, 16, 8)
