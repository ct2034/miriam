import logging
import numpy as np
from bintrees import rbtree

from tools import ColoredLogger

logging.setLoggerClass(ColoredLogger)

def astar_base(start, condition, heuristic, get_children, cost, goal_test):
    _, start = cost(condition, start)  # it may have collisions

    closed = []
    open_tree = rbtree.RBTree()
    g_score = {}
    g_score[start] = 0
    f_score = {}
    f_score[start] = heuristic(condition, start)

    open_tree[f_score[start]] = start

    while len(open_tree) > 0:
        # the node in openSet having the lowest fScore[] value
        current = get_best(open_tree)

        if goal_test(condition, current):
            return current

        remove(open_tree, current, {}, f_score)

        closed.append(current)
        children = get_children(condition, current)
        for neighbor in children:
            if neighbor in closed:
                continue  # Ignore the neighbor which is already evaluated.
            # The distance from start to a neighbor
            c, neighbor = cost(condition, neighbor)
            if c >= 99999:
                closed.append(neighbor)
                continue  # This is not part of a plan

            tentative_g_score = c

            append = True
            if not is_in(open_tree, neighbor, g_score, f_score):  # Discover a new node
                add(open_tree, neighbor, g_score, {})
            elif tentative_g_score >= g_score[neighbor]:
                continue  # This is not a better path.
            else:
                append = False
            g_score[neighbor] = tentative_g_score
            f_score[neighbor] = g_score[neighbor]
            f_score[neighbor] += heuristic(condition, neighbor)
            if append:
                remove(open_tree, current, g_score, {})
                add(open_tree, neighbor, {}, f_score)

    raise RuntimeError("Can not find a solution")


def get_best(open_tree):
    for k in open_tree:
        n = open_tree[k]
        break
    return n


def remove(open_tree, node, f_score, g_score):
    for score in [f_score, g_score]:
        if node in score:
            try:
                open_tree.remove(score[node])
            except KeyError:
                pass


def add(open_tree, node, f_score, g_score):
    for score in [f_score, g_score]:
        if node in score:
            open_tree[score[node]] = node
            break


def is_in(open_tree, node, f_score, g_score):
    for score in [f_score, g_score]:
        if node in score and score[node] in open_tree:
            return True
    return False
