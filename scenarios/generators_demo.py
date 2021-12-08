#!/usr/bin/env python3
from matplotlib import pyplot as plt

from scenarios.generators import (building_walls, movingai,
                                  tracing_pathes_in_the_dark)
from visualization import plot_env_with_arrows


def demo_movingai():
    n_agents = 25
    env, starts, goals = movingai("Paris_1_256", "even", 0, n_agents)
    plot_env_with_arrows(env, starts, goals)


def demo_tracing_pathes_in_the_dark():
    n_agents = 3
    env, starts, goals = tracing_pathes_in_the_dark(50, .5, n_agents, 0)
    plot_env_with_arrows(env, starts, goals)


def demo_building_walls():
    n_agents = 6
    env, starts, goals = building_walls(8, .3, n_agents, 0)
    plot_env_with_arrows(env, starts, goals)
    plt.show()


if __name__ == "__main__":
    demo_building_walls()
