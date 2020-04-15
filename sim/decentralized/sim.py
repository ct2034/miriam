#!/usr/bin/env python3

import random
import itertools

import numpy as np
from matplotlib import pyplot as plt
import networkx as nx


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


def plot(
        environent: np.ndarray, agents: np.ndarray,
        goals: np.ndarray, paths: np.ndarray):
    """Plot the environment map with `x` coordinates to the right, `y` up.
    Occupied by colourful agents."""
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
        ax.plot(a[0], a[1], markersize=5, marker='o')

    # goals
    for i_a, a in enumerate(goals):
        ax.plot(a[0], a[1], markersize=5, marker='x')

    # paths
    for p in paths:
        ax.plot(p[:, 0], p[:, 1])

    plt.show()


def initialize_new_agent(environent, agents) -> np.ndarray:
    """Place new agent in the environment, where no obstacle or other agent
    is."""
    environent_with_agents = environent.copy()
    for a in agents:
        environent_with_agents[tuple(a)] = 1
    no_obstacle_nor_agent = np.where(environent_with_agents == 0)
    gen = np.random.default_rng()
    return gen.choice(no_obstacle_nor_agent, axis=1)


def initialize_agents(environent, n_agents) -> np.ndarray:
    """Initialize `n_agents` many agents in unique, free spaces of
    `environment`, (not colliding with each other)."""
    agents = np.ndarray([0, 2], dtype=np.int64)
    for i_a in range(n_agents):
        agent = initialize_new_agent(environent, agents)
        agents = np.append(agents, [agent], axis=0)
    return agents


def plan_path(
        env_nx: nx.Graph, start: np.ndarray, goal: np.ndarray) -> np.ndarray:
    """Plan a path in the environment"""
    tuple_path = nx.shortest_path(env_nx, tuple(start), tuple(goal))
    return np.array(tuple_path)


def plan_paths(
        env_nx: nx.Graph, starts: np.ndarray, goals: np.ndarray) -> list:
    """Plan paths for all agents between their starts and goals."""
    paths = []
    assert len(starts) == len(goals)
    for i_a in range(len(starts)):
        paths.append(plan_path(env_nx, starts[i_a], goals[i_a]))
    return paths


def prepare_step(pos: np.ndarray, path: np.ndarray) -> np.ndarray:
    """Return the next step, if the agent is currently at `pos`,
    returns end pose if agent reached goal."""
    path_len = len(path)
    i = path.index(pos)
    if i == path_len:
        return path[i]  # agent is at its goal
    else:
        return path[i+1]


def check_for_colissions(
        poses: np.ndarray, next_poses: np.ndarray) -> (dict, dict):
    """check for two agents going to meet at one vertex or two agents using
    the same edge."""
    node_colissions = {}
    edge_colissions = {}
    for i_a in range(len(poses)):
        for i_oa in [i for i in range(len(poses)) if i != i_a]:
            if next_poses[i_a] == next_poses[i_oa]:
                node_colissions[next_poses[i_a]] = [i_a, i_oa]
            if (next_poses[i_a] == poses[i_oa] and
                    poses[i_a] == next_poses[i_oa]):
                edge = [poses[i_a], poses[i_oa]]
                sorted(edge)
                node_colissions[edge] = [i_a, i_oa]
    return node_colissions, edge_colissions


def iterate_sim(agents: np.ndarray, paths: np.ndarray) -> np.ndarray:
    """Given a set of current agent poses, find possible next steps for each
    agent."""
    possible_next_agent_poses = agents.copy()
    next_agent_poses = agents.copy()
    # prepare step
    for i_a in range(len(agents)):
        possible_next_agent_poses[i_a] = prepare_step(agents[i_a], paths[i_a])
    # check collisions
    node_colissions, edge_colissions = check_for_colissions(
        agents, possible_next_agent_poses)

    return next_agent_poses


if __name__ == "__main__":
    n_agents = 10

    # maze (environment)
    env = initialize_environment(10, .1)
    env_nx = gridmap_to_nx(env)

    # agents
    agents = initialize_agents(env, n_agents)
    goals = initialize_agents(env, n_agents)

    # paths
    paths = plan_paths(env_nx, agents, goals)

    # display
    plot(env, agents, goals, paths)
