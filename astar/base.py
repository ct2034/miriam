import numpy as np

"""My A* Planner
based on: https://en.wikipedia.org/wiki/A*_search_algorithm#Pseudocode"""
def astar_base(start, goal, map, heuristic, reconstruct_path, get_children, cost):
    # config
    # """Time Dimension Size"""
    # time_len = heuristic(start, goal, map) * 20
    # ------

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

    raise RuntimeError("Can not find a path")


def min_f_open(open, f_score):
    current_min_val = np.Inf
    current_min_index = 0
    for o in open:
        if f_score[o] < current_min_val:
            current_min_val = f_score[o]
            current_min_index = o
    return current_min_index
