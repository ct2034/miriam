#!/usr/bin/env python3

from matplotlib import pyplot as plt
import numpy as np

from scenarios.generators import movingai

if __name__ == "__main__":
    n_agents = 25
    grid, starts, goals = movingai("Paris_1_256", "even", 0, n_agents)

    plt.imshow(np.swapaxes(grid, 0, 1), cmap='Greys', origin='lower')
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
