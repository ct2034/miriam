#!/usr/bin/env python3

import random
import itertools

import numpy as np
from matplotlib import pyplot as plt


def initialize_environment(size, fill):
    """Make a square map with edge length `size` and
    `fill` (0..1) obstacle ratio."""
    environent = np.zeros([size, size], dtype=np.int64)
    n_to_fill = int(fill * size ** 2)
    to_fill = random.sample(
        list(itertools.product(range(size), repeat=2)), k=n_to_fill)
    for cell in to_fill:
        environent[cell] = 1
    return environent


def plot(environent, agents):
    """Plot the environment map with `x` coordinates to the right, `y` up.
    Occupied by colourful agents."""
    # map
    image = environent * -.5 + 1
    image = np.swapaxes(image, 0, 1)
    fig, ax = plt.subplots()
    c = ax.pcolor(image, edgecolors='k', linewidths=.5,
                  linestyle=':', cmap='gray', vmin=0, vmax=1)
    ax.set_aspect('equal')
    baserange = np.arange(environent.shape[0], step=2)
    ax.set_xticks(baserange + .5)
    ax.set_xticklabels(map(str, baserange))
    ax.set_yticks(baserange + .5)
    ax.set_yticklabels(map(str, baserange))

    # agents
    for i_a, a in enumerate(agents):
        ax.plot(a[0]+.5, a[1]+.5, markersize=5, marker='o')

    plt.show()


def initialize_new_agent(environent, agents):
    """Place new agent in the environment, where no obstacle or other agent
    is."""
    environent_with_agents = environent.copy()
    for a in agents:
        environent_with_agents[tuple(a)] = 1
    no_obstacle_nor_agent = np.where(environent_with_agents == 0)
    gen = np.random.default_rng()
    return gen.choice(no_obstacle_nor_agent, axis=1)


def initialize_agents(environent, n_agents):
    """Initialize `n_agents` many agents in unique, free spaces of
    `environment`, (not colliding with each other)."""
    agents = np.ndarray([0, 2], dtype=np.int64)
    for i_a in range(n_agents):
        agent = initialize_new_agent(environent, agents)
        agents = np.append(agents, [agent], axis=0)
    return agents


if __name__ == "__main__":
    # maze (environment)
    env = initialize_environment(10, .2)

    # agents
    agents = initialize_agents(env, 10)

    # display
    plot(env, agents)
