#!/usr/bin/env python3

import random

import matplotlib.pyplot as plt
import numpy as np

from plan_ecbs import plan_in_gridmap


def make_random_gridmap(width, height, fill) -> np.ndarray:
    gridmap = np.zeros((width, height))
    obstacle_cells = int(fill * width * height)
    for _ in range(obstacle_cells):
        gridmap[
            random.randint(0, width-1),
            random.randint(0, height-1)
        ] = 1
    return gridmap


def show_map(x):
    plt.imshow(
        np.swapaxes(x, 0, 1),
        aspect='equal',
        cmap=plt.get_cmap("binary"),
        origin='lower')
    plt.show()


def is_free(gridmap, pos):
    return gridmap[tuple(pos)] == 0


def get_random_free_pos(gridmap, width, height):
    def random_pos(width, height):
        return [
            random.randint(0, width-1),
            random.randint(0, height-1)
        ]
    pos = random_pos(width, height)
    while not is_free(gridmap, pos):
        pos = random_pos(width, height)
    return pos


if __name__ == "__main__":
    width = 10
    height = 10
    random.seed(1)
    n_agents = 3
    for n in range(10):
        gridmap = make_random_gridmap(width, height, .3)
        starts = [get_random_free_pos(gridmap, width, height)
                  for _ in range(n_agents)]
        goals = [get_random_free_pos(gridmap, width, height)
                 for _ in range(n_agents)]
        plan_in_gridmap(gridmap, starts, goals)
