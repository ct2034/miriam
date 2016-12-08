import timeit

import numpy as np

from astar import astar_grid8con

grid = np.zeros([10, 10])


def test_get_children_corner():
    children = astar_grid8con.get_children((0, 0), grid)
    for child in [(0, 1), (1, 1), (1, 0)]:
        children.remove(child)
    assert len(children) == 0, "Not getting all the children"


def test_get_children_middle():
    children = astar_grid8con.get_children((2, 2), grid)
    for child in [(2, 1), (1, 1), (1, 2), (1, 3), (2, 3), (3, 3), (3, 2), (3, 1)]:
        children.remove(child)
    assert len(children) == 0, "Not getting all the children"


def test_astar():
    grid[:8, 2] = -1
    grid[8, 2:6] = -1
    grid[2, 4:8] = -1
    grid[2:, 8] = -1

    # timing ..
    measurements = 10

    t = timeit.Timer()

    for i in range(measurements):
        t.timeit()
        path = astar_grid8con.astar_grid8con((1, 1), (9, 9), grid)

        assert astar_grid8con.path_length(path) < 33.8994949367

    print("duration mean:", np.mean(t.repeat()), "s, std:", np.std(t.repeat()))

    assert np.mean(t.repeat()) < .05
