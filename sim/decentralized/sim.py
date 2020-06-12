#!/usr/bin/env python3

import itertools
import random
from typing import List

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

from agent import Agent, Policy


def initialize_environment(size: int, fill: float):
    """Make a square map with edge length `size` and
    `fill` (0..1) obstacle ratio."""
    environent = np.zeros([size, size], dtype=np.int64)
    n_to_fill = int(fill * size ** 2)
    to_fill = random.sample(
        list(itertools.product(range(size), repeat=2)), k=n_to_fill)
    for cell in to_fill:
        environent[cell] = 1
    return environent


def gridmap_to_nx(env: np.ndarray):
    """convert numpy gridmap into networkx graph."""
    g = nx.grid_graph(dim=list(env.shape))
    obstacles = np.where(env == 1)
    for i_o in range(len(obstacles[0])):
        g.remove_node(
            (obstacles[0][i_o],
             obstacles[1][i_o])
        )
    return g


def initialize_new_agent(
        env: np.ndarray, env_nx: nx.Graph, agents: List[Agent], policy: Policy
) -> List[Agent]:
    """Place new agent in the environment, where no obstacle or other agent
    is."""
    env_with_agents = env.copy()
    env_with_goals = env.copy()
    for a in agents:
        env_with_agents[tuple(a.pos)] = 1
        env_with_goals[tuple(a.goal)] = 1
    no_obstacle_nor_agent = np.where(env_with_agents == 0)
    no_obstacle_nor_goal = np.where(env_with_goals == 0)
    gen = np.random.default_rng()
    pos = gen.choice(no_obstacle_nor_agent, axis=1)
    goal = gen.choice(no_obstacle_nor_goal, axis=1)
    a = Agent(env, env_nx, pos, policy)
    a.give_a_goal(goal)
    return a


def initialize_agents(
        env: np.ndarray, env_nx: nx.Graph, n_agents: int, policy: Policy
) -> List[Agent]:
    """Initialize `n_agents` many agents in unique, free spaces of
    `environment`, (not colliding with each other)."""
    agents = []
    for i_a in range(n_agents):
        agent = initialize_new_agent(env, env_nx, agents, policy)
        agents.append(agent)
    return agents


def get_possible_next_agent_poses(
        agents: List[Agent],
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
        agents: List[Agent],
        nxt_poses: np.ndarray) -> (dict, dict):
    """check for two agents going to meet at one vertex or two agents using
    the same edge."""
    node_colissions = {}
    edge_colissions = {}
    # ignoring finished agents
    for i_a in range(len(agents)):
        if not agents[i_a].is_at_goal():
            for i_oa in [i for i in range(len(agents)) if i != i_a]:
                if not agents[i_oa].is_at_goal():
                    if (nxt_poses[i_a] ==
                            nxt_poses[i_oa]).all():
                        node_colissions[tuple(nxt_poses[i_a])] = [i_a, i_oa]
                    if ((nxt_poses[i_a] == agents[i_oa].pos).all() and
                            (nxt_poses[i_oa] == agents[i_a].pos).all()):
                        edge = [tuple(nxt_poses[i_a]),
                                tuple(agents[i_a].pos)]
                        edge_colissions[tuple(sorted(edge))] = [i_a, i_oa]
    return node_colissions, edge_colissions


def iterate_sim(agents: List[Agent]):
    """Given a set of agents, find possible next steps for each
    agent and move them there if possible."""
    can_proceed = [True] * len(agents)
    there_are_collisions = True

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
                if agents[i_a1].get_priority() > agents[i_a2].get_priority():
                    can_proceed[i_a2] = False  # has lower prio
                else:
                    can_proceed[i_a1] = False  # has lower prio
            for edge, [i_a1, i_a2] in edge_colissions.items():
                if agents[i_a1].get_priority() > agents[i_a2].get_priority():
                    can_proceed[i_a2] = False  # has lower prio
                else:
                    can_proceed[i_a1] = False  # has lower prio

        if not any(can_proceed):
            raise Exception("Deadlock")

    for i_a in range(len(agents)):
        if can_proceed[i_a]:
            agents[i_a].make_next_step(possible_next_agent_poses[i_a, :])


def are_all_agents_at_their_goals(agents: List[Agent]) -> bool:
    """Returns true iff all agents are at their respective goals."""
    return all(map(lambda a: a.is_at_goal(), agents))


def plot(environent: np.ndarray, agents: np.ndarray):
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


if __name__ == "__main__":
    n_agents = 10

    # maze (environment)
    env = initialize_environment(10, .1)
    env_nx = gridmap_to_nx(env)

    # agents
    agents = initialize_agents(env, env_nx, n_agents, Policy.RANDOM)

    # iterate
    while not are_all_agents_at_their_goals(agents):
        plot(env, agents)
        iterate_sim(agents)
