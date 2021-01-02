#!/usr/bin/env python3

from matplotlib import pyplot as plt
import numpy as np

from scenarios.generators import movingai, tracing_pathes_in_the_dark


def demo_movingai():
    n_agents = 25
    env, starts, goals = movingai("Paris_1_256", "even", 0, n_agents)
    plot(env, starts, goals)


def plot(env, starts, goals):
    plt.imshow(np.swapaxes(env, 0, 1), cmap='Greys', origin='lower')
    n_agents = len(starts)
    for i_a in range(n_agents):
        plt.arrow(
            starts[i_a][0] + .5,
            starts[i_a][1] + .5,
            goals[i_a][0] - starts[i_a][0],
            goals[i_a][1] - starts[i_a][1],
            width=1,
            length_includes_head=True,
            linewidth=0
        )

    plt.show()


def demo_tracing_pathes_in_the_dark():
    n_agents = 3
    env, starts, goals = tracing_pathes_in_the_dark(50, .5, n_agents, 0)
    plot(env, starts, goals)


if __name__ == "__main__":
    demo_tracing_pathes_in_the_dark()
