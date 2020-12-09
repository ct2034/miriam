#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
import libastar
import sys
from itertools import product

np.random.seed(0)
grid = np.random.random_integers(0, 10, size=(100, 50))
plt.imshow(grid)
plt.show()

maze = libastar.Maze(100, 50)
maze.set_goal(maze.get_vertex(99, 49))
for x, y in product(range(100), range(50)):
    if grid[x, y] > 8:
        maze.add_barrier(maze.get_vertex(x, y))
print(maze.solve())
print(maze.to_string())
