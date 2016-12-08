import numpy as np

from smartleitstand import astar

grid = np.zeros([4, 4])


def test_get_children_corner():
    children = astar.get_children((0, 0), grid)
    for child in [(0, 1), (1, 1), (1, 0)]:
        children.remove(child)
    assert len(children) == 0, "We have removed it all"


def test_get_children_middle():
    children = astar.get_children((2, 2), grid)
    for child in [(2, 1), (1, 1), (1, 2), (1, 3), (2, 3), (3, 3), (3, 2), (3, 1)]:
        children.remove(child)
    assert len(children) == 0, "We have removed it all"


def test_astar():
    map = np.zeros([10, 10])
    map[:8, 2] = -1
    map[8, 2:6] = -1
    map[2, 4:8] = -1
    map[2:, 8] = -1

    path = astar.astar((1, 1), (9, 9), map)

    assert astar.path_length(path) < 33.8994949367
