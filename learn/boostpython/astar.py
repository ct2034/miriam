#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
import libastar
import sys
from itertools import product

width = 100
height = 50

grid = np.random.random_integers(0, 10, size=(width, height))
plt.imshow(grid)

maze = libastar.Maze(width, height, 0, 0, 99, 49)
for x, y in product(range(width), range(height)):
    if grid[x, y] > 9:
        if not maze.add_barrier(x, y):
            print("Failed to add base obstacle {}, {}".format(x, y))
for x in [3, 10, 15, 23, 44, 53, 77, 80, 88, 99]:
    for y in range(height):
        if grid[x, y] > 3:
            if not maze.add_barrier(x, y):
                print("Failed to add additional obstacle {}, {}".format(x, y))

print(maze.solve())
print(maze.to_string())
