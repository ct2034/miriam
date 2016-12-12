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
    """Should we have a 0-connected graph (else 4)"""
    eight_con = False

    if eight_con:
        ds = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        raise NotImplementedError("8-connected not in spacetime")
    else:
        ds = [(-1, 0, 1), (0, -1, 1), (0, 1, 1), (1, 0, 1), (0, 0, 1)]  # ugly but faster

    _children = []
    for d in ds:
        checking = (current[0] + d[0], current[1] + d[1], current[2] + d[2])
        try:
            if np.min(checking[0:2]) >= 0:  # no negative coord to not wrap
                grid_checking = grid[checking]
                if grid_checking >= 0:  # no obstacle
                    _children.append(checking)
        except IndexError:
            pass
    return _children


def heuristic(a, b, grid: np.array = False):
    if grid:  # costmap values
        h = np.mean(grid[grid >= 0]) * distance(a, b)
    else:
        h = distance(a, b)
    assert h >= 0, "Negative Heuristic"
    return h


def cost(a, b, grid=False):
    if grid:  # costmap values
        return grid[a] * distance_grid(a, b)
    else:
        return distance_grid(a, b)


def distance(a: tuple, b: tuple) -> float:
    space_dist = np.abs(a[0] - b[0]) + np.abs(a[1] - b[1])  # Manhattan Distance
    if space_dist == 0:
        return 1.9  # waiting is a bit cheaper then driving twice
    else:
        return space_dist


def distance_grid(a, b):
    dist_tuple = (a[0] - b[0], a[1] - b[1])
    if ((dist_tuple == (1, 0)) |
            (dist_tuple == (-1, 0)) |
            (dist_tuple == (0, 1)) |
            (dist_tuple == (0, -1))
        ):
        return 1
    elif dist_tuple == (0, 0):
        return 1.9  # waiting is a bit cheaper then driving twice
    else:
        raise ArithmeticError("Unknown Distance")


def path_length(path: list) -> int:
    """Path length calculation (for eval only)"""
    l = 0
    for i in range(len(path)):
        if i != 0:
            l += distance_grid(path[i - 1], path[i])
    return l


def astar_grid4con(start, goal, map):
    return base.astar_base(start, goal, map, heuristic, reconstruct_path, get_children, cost)


def plot(path, map):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    zz = map[:, :, 0].T
    xx, yy = np.meshgrid(np.arange(len(zz)),
                         np.arange(len(zz)))
    ax.contourf(xx, yy, zz, cmap='Greys', zdir='z', alpha=.2)

    patharray = np.array(path)
    ax.plot(
        xs=patharray[:, 0],
        ys=patharray[:, 1],
        zs=patharray[:, 2],
        c='b'
    )

    plt.title('Astar path')
    ax.axis([0, map.shape[0], 0, map.shape[1]])

    plt.show()

