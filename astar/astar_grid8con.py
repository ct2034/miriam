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
    _children = []
    for d in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:  # ugly but faster
        checking = (d[0] + current[0], d[1] + current[1])  # ugly but faster
        try:
            if np.min(checking) >= 0:  # no negative coord to not wrap
                grid_checking = grid[checking]
                if grid_checking >= 0:  # no obstacle
                    _children.append(checking)
        except IndexError:
            pass
    return _children


def heuristic(a, b, grid):
    return np.mean(abs(grid)) * distance(a, b)


def cost(a, b, grid):
    if np.max(grid) > 0:
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
