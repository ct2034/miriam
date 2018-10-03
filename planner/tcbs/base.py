import logging
import numpy as np
import datetime

from tools import ColoredLogger

logging.setLoggerClass(ColoredLogger)

def astar_base(start, condition, heuristic, get_children, cost, goal_test):
    _, start = cost(condition, start)  # it may have collisions

    collect_stats = False
    stats = {}

    closed = []
    open = [start]
    g_score = {}
    g_score[start] = 0
    f_score = {}
    f_score[start] = heuristic(condition, start)

    f_score_open = np.array([])
    f_score_open = np.append(f_score_open, f_score[start])

    while len(open) > 0:
        # the node in openSet having the lowest fScore[] value
        current = argmin_f_open(open, f_score_open)

        if goal_test(condition, current):
            return current

        i_rm = open.index(current)
        open.remove(current)
        f_score_open = np.delete(f_score_open, i_rm)

        closed.append(current)
        children = get_children(condition, current)
        for neighbor in children:

            if neighbor in closed:
                continue  # Ignore the neighbor which is already evaluated.
            # The distance from start to a neighbor
            c, neighbor = cost(condition, neighbor)

            if collect_stats:
                stats[datetime.datetime.now().microsecond] = {
                    "closed.__len__()": closed.__len__(),
                    "open.__len__()": open.__len__(),
                    "neighbor[2].__len__() (blocks)": neighbor[2].__len__(),
                    "c": c
                }

            if c >= 99999:
                closed.append(neighbor)
                continue  # This is not part of a plan

            tentative_g_score = c

            append = True
            if neighbor not in open:  # Discover a new node
                open.append(neighbor)
            elif tentative_g_score >= g_score[neighbor]:
                continue  # This is not a better path.
            else:
                append = False
            g_score[neighbor] = tentative_g_score
            the_f_score = tentative_g_score + heuristic(condition, neighbor)
            f_score[neighbor] = the_f_score
            if append:
                f_score_open = np.append(f_score_open, the_f_score)

    raise RuntimeError("Can not find a solution")


def argmin_f_open(open_list, f_score_open):
    assert len(open_list) == len(f_score_open), "Lengths must be equal"
    return open_list[np.argmin(f_score_open)]
