import numpy as np


def astar(start, goal, map):
    """based on: https://en.wikipedia.org/wiki/A*_search_algorithm#Pseudocode"""
    # The set of nodes already evaluated.
    closed = []
    # The set of currently discovered nodes still to be evaluated.
    # Initially, only the start node is known.
    open = [start]
    # For each node, which node it can most efficiently be reached from.
    # If a node can be reached from many nodes, cameFrom will eventually contain the
    # most efficient previous step.
    cameFrom = {}

    # For each node, the cost of getting from the start node to that node.
    g_score = np.full(map.shape, np.Inf)
    # The cost of going from start to start is zero.
    g_score[start] = 0
    # For each node, the total cost of getting from the start node to the goal
    # by passing by that node. That value is partly known, partly heuristic.
    f_score = np.full(map.shape, np.Inf)
    # For the first node, that value is completely heuristic.
    f_score[start] = heuristic(start, goal, map)

    while len(open) > 0:
        current = min_f_open(open, f_score)  # the node in openSet having the lowest fScore[] value

        if current == goal:
            return reconstruct_path(cameFrom, current)

        open.remove(current)
        closed.append(current)
        children = get_children(current, map)
        for neighbor in children:
            if neighbor in closed:
                continue  # Ignore the neighbor which is already evaluated.
            # The distance from start to a neighbor
            tentative_g_score = g_score[current] + cost(current, neighbor, map)
            if neighbor not in open:  # Discover a new node
                open.append(neighbor)
            elif tentative_g_score >= g_score[neighbor]:
                continue  # This is not a better path.

            # This path is the best until now. Record it!
            cameFrom[neighbor] = current
            g_score[neighbor] = tentative_g_score
            f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal, map)

    return -1  # failure


def reconstruct_path(came_from, current):
    total_path = [current]
    while current in came_from.keys():
        current = came_from[current]
        total_path.append(current)
    total_path.reverse()
    return total_path


def get_children(current, grid):
    __children = []
    # for d in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
    for d in [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]:
        checking = np.array(current) + np.array(d)
        if (
            (checking[0] < grid.shape[0]) &  # bounds ->
            (checking[1] < grid.shape[1]) &  # bounds ->|
            (checking[0] >= 0) &  # bounds  <-
            (checking[1] >= 0)  # bounds |<-
        ):
            if grid[tuple(checking)] >= 0:  # free
                __children.append(tuple(checking))
    return __children


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


def min_f_open(open, f_score):
    current_min_val = np.Inf
    current_min_index = 0
    for o in open:
        if f_score[o] < current_min_val:
            current_min_val = f_score[o]
            current_min_index = o
    return current_min_index
