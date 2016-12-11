import datetime
import numpy as np

from astar import astar_grid48con


def rand_coords(x, y):
    return (np.random.randint(0, x),
            np.random.randint(0, y))

"""Deterministic test?"""
det = False

if det:
    """Scale of map (min: 1)"""
    s = 2

    _grid = np.zeros([s * 10, s * 10, s * 50])
    _grid[:s * 8, s * 2, :] = -1
    _grid[s * 8, s * 2:s * 6, :] = -1
    _grid[s * 2, s * 4:s * 8, :] = -1
    _grid[s * 2:, s * 8, :] = -1

    start = (s * 1, s * 1, 0)
    goal = (s * 9, s * 9, s * 49)
else:
    size = 100
    _grid = np.zeros([size, size, size * 2])
    for i in range(int(np.round(pow(size, 2) * .3))):
        _grid[rand_coords(size, size)] = -1

    start = (rand_coords(size, size) + (0,))
    while _grid[start] == -1:
        start = (rand_coords(size, size) + (0,))

    goal = (rand_coords(size, size) + (size * 2 - 1,))
    while _grid[goal] == -1:
        goal = (rand_coords(size, size) + (size * 2 - 1,))

startt = datetime.datetime.now()

path = astar_grid48con.astar_grid8con(start, goal, _grid)

print("computation time:", (datetime.datetime.now() - startt).total_seconds(), "s")

print("length: ", astar_grid48con.path_length(path))

astar_grid48con.plot(path, _grid)
