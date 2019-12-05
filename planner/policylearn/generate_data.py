#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import random

import bench


def make_random_gridmap(width, height, fill):
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
    return gridmap[pos] == 0


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


gridmap = make_random_gridmap(20, 10, .1)
