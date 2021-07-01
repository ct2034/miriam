#!/usr/bin/env python3
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from scenarios.visualization import plot_with_paths

import networkx as nx
from networkx.classes.graphviews import subgraph_view
from networkx.convert import to_dict_of_dicts


def env_to_nx(env):
    t = env.shape[0] * env.shape[1]
    dim = (t,) + env.shape
    g = nx.DiGraph(nx.grid_graph(dim, periodic=False))
    free = np.min(env)

    for i_t in range(t-1):
        t_from = i_t
        t_to = i_t + 1
        for x, y in product(range(env.shape[0]), range(env.shape[1])):
            for x_to in [x-1, x+1]:
                n_to = (x_to, y, t_to)
                if n_to in g.nodes():
                    g.add_edge((x, y, t_from), n_to)
            for y_to in [y-1, y+1]:
                n_to = (x, y_to, t_to)
                if n_to in g.nodes():
                    g.add_edge((x, y, t_from), n_to)

    def filter_node(n):
        return env[n[0], n[1]] == free

    def filter_edge(n1, n2):
        return n2[2] > n1[2]
    return nx.subgraph_view(g, filter_node, filter_edge)


def plan_timed(g, start, goal, blocks):
    print(f"start: {start}")
    print(f"goal: {goal}")
    blocks = set(map(lambda b: tuple(b), blocks))

    def cost(e):
        if (
            e[0][0] == goal[0] and
            e[0][1] == goal[1] and
            e[1][0] == goal[0] and
            e[1][1] == goal[1]
        ):
            # waiting at goal is free
            return 0
        if (
            e[0][0] == e[1][0] and
            e[0][1] == e[1][1]
        ):
            # waiting generally is a little cheaper
            return .9999
        else:
            # normal cost
            return 1

    nx.set_edge_attributes(g, {e: cost(e) for e in g.edges()}, "cost")

    def filter_node(n):
        return n not in blocks

    g_blocks = nx.subgraph_view(g, filter_node=filter_node)

    t_max = np.max(np.array(g.nodes())[:, 2])

    def dist(a, b):
        (x1, y1, _) = a
        (x2, y2, _) = b
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    p = np.array(nx.astar_path(
        g_blocks,
        start + (0,),
        goal + (t_max,),
        heuristic=dist,
        weight="cost"))
    end = None
    assert all(p[-1][:2] == goal)
    i = len(p) - 1
    while i > 0 or end is None:
        if all(p[i][:2] == goal):
            end = i+1
        i -= 1
    assert end is not None
    return p[0:end]


if __name__ == "__main__":
    size = 10
    env = np.array(np.random.random((size, size)) < .1, dtype=int)
    g = env_to_nx(env)

    start = (
        np.where(env == 0)[0][0],
        np.where(env == 0)[1][0])
    goal = (
        np.where(env == 0)[0][-1],
        np.where(env == 0)[1][-1])

    blocks = []
    paths = []
    success = True
    while success:
        try:
            path = plan_timed(g, start, goal, blocks)
            paths.append(path)
            i_r = np.random.randint(1, len(path)-1)
            blocks.append(path[i_r])
        except nx.NetworkXNoPath as e:
            print(e)
            success = False

    plot_with_paths(env, paths)
    plt.show()
