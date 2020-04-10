#!/usr/bin/env python3

import random
import itertools

import numpy as np
from matplotlib import pyplot as plt


def initialize_environment(size, fill):
    """Make a square map with edge length `size` and
    `fill` (0..1) obstacle ratio."""
    environent = np.zeros([size, size])
    n_to_fill = int(fill * size ** 2)
    to_fill = random.sample(
        list(itertools.product(range(size), repeat=2)), k=n_to_fill)
    for cell in to_fill:
        environent[cell] = 1
    return environent


def plot(environent):
    """Plot the environment map with `x` coordinates to the right, `y` up."""
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
    plt.show()


if __name__ == "__main__":
    env = initialize_environment(10, .2)
    plot(env)
