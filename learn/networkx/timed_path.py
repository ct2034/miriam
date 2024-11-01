#!/usr/bin/env python3
from itertools import product
from typing import Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from scenarios.visualization import plot_with_paths


def env_to_nx(env):
    t = env.shape[0] * env.shape[1]
    dim = (t,) + env.shape
    g = nx.DiGraph(nx.grid_graph(dim, periodic=False))
    free = np.min(env)

    for i_t in range(t - 1):
        t_from = i_t
        t_to = i_t + 1
        for x, y in product(range(env.shape[0]), range(env.shape[1])):
            for x_to in [x - 1, x + 1]:
                n_to = (x_to, y, t_to)
                if n_to in g.nodes():
                    g.add_edge((x, y, t_from), n_to)
            for y_to in [y - 1, y + 1]:
                n_to = (x, y_to, t_to)
                if n_to in g.nodes():
                    g.add_edge((x, y, t_from), n_to)

    def filter_node(n):
        return env[n[0], n[1]] == free

    def filter_edge(n1, n2):
        return n2[2] > n1[2]

    return nx.subgraph_view(g, filter_node, filter_edge)


def plan_timed(
    g: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    node_blocks: Set[Tuple[int, int, int]],
    edge_blocks: Set[Tuple[Tuple[int, int], Tuple[int, int], int]],
):
    print(f"start: {start}")
    print(f"goal: {goal}")
    print(f"node_blocks: {node_blocks}")
    print(f"edge_blocks: {edge_blocks}")

    def cost(e):
        if (
            e[0][0] == goal[0]
            and e[0][1] == goal[1]
            and e[1][0] == goal[0]
            and e[1][1] == goal[1]
        ):
            # waiting at goal is free
            return 0
        if e[0][0] == e[1][0] and e[0][1] == e[1][1]:
            # waiting generally is a little cheaper
            return 1.0 - 1e-9
        else:
            # normal cost
            return 1

    nx.set_edge_attributes(g, {e: cost(e) for e in g.edges()}, "cost")

    def filter_node(n):
        return n not in node_blocks

    def filter_edge(n1, n2):
        return (n1[:2], n2[:2], n1[2]) not in edge_blocks and (
            n2[:2],
            n1[:2],
            n1[2],
        ) not in edge_blocks

    g_blocks = nx.subgraph_view(g, filter_node=filter_node, filter_edge=filter_edge)

    t_max = np.max(np.array(g.nodes())[:, 2])

    def dist(a, b):
        (x1, y1, _) = a
        (x2, y2, _) = b
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    p = np.array(
        nx.astar_path(
            g_blocks, start + (0,), goal + (t_max,), heuristic=dist, weight="cost"
        )
    )
    end = None
    assert all(p[-1][:2] == goal)
    i = len(p) - 1
    while i > 0 or end is None:
        if all(p[i][:2] == goal):
            end = i + 1
        i -= 1
    assert end is not None
    return p[0:end]


if __name__ == "__main__":
    np.random.seed(0)
    size = 8
    env = np.array(np.random.random((size, size)) < 0.2, dtype=int)
    g = env_to_nx(env)

    node_blocks: Set[Tuple[int, int, int]] = set()
    edge_blocks: Set[Tuple[Tuple[int, int], Tuple[int, int], int]] = set()
    paths = []
    i = 0
    while i < 16:
        it = np.random.randint(np.count_nonzero(env == 0))
        ig = np.random.randint(np.count_nonzero(env == 0))
        start = (np.where(env == 0)[0][it], np.where(env == 0)[1][it])
        goal = (np.where(env == 0)[0][ig], np.where(env == 0)[1][ig])
        try:
            path = plan_timed(g, start, goal, node_blocks, edge_blocks)
            paths.append(path)
            for ip in range(len(path)):
                node_blocks.add((path[ip, 0], path[ip, 1], path[ip, 2]))
                if ip < len(path) - 1:
                    edge_blocks.add(
                        (tuple(path[ip, :2]), tuple(path[ip + 1, :2]), int(path[ip, 2]))
                    )
        except nx.NetworkXNoPath as e:
            print(e)
        except nx.NodeNotFound as e:
            print(e)
        i += 1

    plot_with_paths(env, paths)
    plt.show()
