"""
=========================
Simple animation examples
=========================

This example contains two animations. The first is a random walk plot. The
second is an image animation.
"""

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def update_line(num, data, lines):
    for i in range(data.shape[0]):
        lines[i].set_data(data[i, ..., :num])
    return lines


if __name__ == "__main__":
    fig1 = plt.figure()
    n_agents = 4
    data = np.random.rand(n_agents, 2, 10)
    l = [None] * n_agents
    for i in range(n_agents):
        (l[i],) = plt.plot([], [], "-")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("x")
    plt.title("test")
    line_ani = animation.FuncAnimation(
        fig1, update_line, 10, fargs=(data, l), interval=50, blit=True
    )

    # line_ani.save('demo.mp4')

    plt.show()
