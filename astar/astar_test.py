import datetime

import numpy as np

from astar import astar_grid48con

grid = np.zeros([10, 10, 10])


def test_get_children_corner():
    children = astar_grid48con.get_children((0, 0, 0), grid)
    for child in [(0, 1, 1), (1, 0, 1), (0, 0, 1)]:
        children.remove(child)
    assert len(children) == 0, "Not getting all the children"


def test_get_children_middle():
    children = astar_grid48con.get_children((2, 2, 0), grid)
    for child in [(2, 1, 1), (1, 2, 1), (2, 3, 1), (3, 2, 1), (2, 2, 1)]:
        children.remove(child)
    assert len(children) == 0, "Not getting all the children"


def rand_coords(x, y):
    return (np.random.randint(0, x),
            np.random.randint(0, y))


def astar_wait():
    size = 10
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

    try:
        path = astar_grid48con.astar_grid8con(start, goal, _grid)
    except RuntimeError as e:
        print("Could not find path")
        return 0, (datetime.datetime.now() - startt).total_seconds()

    t = (datetime.datetime.now() - startt).total_seconds()

    for i in range(len(path) - 1):
        d = np.array(path[i]) - np.array(path[i + 1])
        assert not (d[0] == 0 and d[1] == 0), "Waiting in place is baaaaad"

    return astar_grid48con.path_length(path), t


def test_astar_wait():
    res = []
    for i in range(10):
        res.append(astar_wait())

    res = np.array(res)
    print("Duration mean:", np.mean(res[res[:, 1] > 0, 1]))
    print("Length mean:", np.mean(res[:, 0]))
