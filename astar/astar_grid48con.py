import numpy as np

from astar import base


def reconstruct_path(came_from, current):
    total_path = [current]
    while current in came_from.keys():
        current = came_from[current]
        total_path.append(current)
    total_path.reverse()
    return total_path


def get_children(current, grid):
    # config
    """Should we have a 0-connected graph (else 4)"""
    eight_con = False
    # ------

    if eight_con:
        ds = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        raise NotImplementedError("8-connected not in spacetime")
    else:
        ds = [(-1, 0, 1), (0, -1, 1), (0, 1, 1), (1, 0, 1), (0, 0, 1)]  # ugly but faster

    _children = []
    for d in ds:
        checking = tuple(np.add(current, d))
        try:
            if np.min(checking) >= 0:  # no negative coord to not wrap
                grid_checking = grid[checking]
                if grid_checking >= 0:  # no obstacle
                    _children.append(checking)
        except IndexError:
            pass
    return _children


def heuristic(a, b, grid):
    # optimistic heuristic takes only distance in space
    return np.mean(abs(grid)) * distance(a[0:2], b[0:2])


def cost(a, b, grid):
    if np.max(grid) > 0:  # costmap values
        return grid[a] * distance(a, b)
    else:
        return distance(a, b)


def distance(a, b):
    return np.linalg.norm(np.array(a, dtype=float) - np.array(b, dtype=float))


def path_length(path):
    """Path length calculation (for eval only)"""
    previous = False
    l = 0
    for p in path:
        if previous:
            l += distance(p, previous)
        previous = p
    return l


def astar_grid8con(start, goal, map):
    return base.astar_base(start, goal, map, heuristic, reconstruct_path, get_children, cost)
