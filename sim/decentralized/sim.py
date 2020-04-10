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


def plot(environent: np.ndarray, agents: np.ndarray):
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


def plan_path(env_nx: nx.Graph, start: np.ndarray, goal: np.ndarray) -> np.ndarray:
    """Plan a path in the environment"""
    return nx.shortest_path(env_nx, tuple(start), tuple(goal))


if __name__ == "__main__":
    # maze (environment)
    env = initialize_environment(10, .1)
    env_nx = gridmap_to_nx(env)

    # agents
    agents = initialize_agents(env, 10)
    goals = initialize_agents(env, 10)

    print(plan_path(env_nx, agents[0], goals[0]))

    # display
    plot(env, agents)
