#!/usr/bin/env python3

import math
import random
import time

import networkx as nx
from libastar_graph import AstarSolver
from matplotlib import pyplot as plt

if __name__ == "__main__":
    g = nx.random_geometric_graph(10000, 0.013)  # type: nx.Graph
    pos = nx.get_node_attributes(g, 'pos')
    posl = [[pos[v][0], pos[v][1]] for v in g.nodes()]
    edges = [[e[0], e[1]] for e in g.edges()]

    a = AstarSolver(posl, edges)
    print(a)
    print(a.retreive(random.randrange(g.number_of_nodes())))
    print(a.retreive(random.randrange(g.number_of_nodes())))
    print(a.retreive(random.randrange(g.number_of_nodes())))
    print("#"*20)

    # assigning edge weights
    for e in g.edges():
        g[e[0]][e[1]]['weight'] = math.sqrt(
            sum([(pos[e[0]][i] - pos[e[1]][i]) ** 2 for i in range(2)]))

    def heuristic(a, b):
        return math.sqrt(sum([(pos[a][i] - pos[b][i]) ** 2 for i in range(2)]))

    # evaluating boost vs networkx
    n_eval = 100
    durations_boost = []
    durations_networkx = []
    for _ in range(n_eval):
        start = random.randrange(g.number_of_nodes())
        end = random.randrange(g.number_of_nodes())

        start_time = time.process_time()
        last_path_boost = a.plan(start, end)
        if len(last_path_boost) > 0:
            durations_boost.append(time.process_time() - start_time)

            start_time = time.process_time()
            last_path_nx = nx.astar_path(
                g, start, end, weight='weight', heuristic=heuristic)
            durations_networkx.append(time.process_time() - start_time)

    print(f"Boost:    {sum(durations_boost)/n_eval:.6f}s")
    print(f"Networkx: {sum(durations_networkx)/n_eval:.6f}s")
    print(f"Boost/Nx: {sum(durations_boost)/sum(durations_networkx):.6f}")
    print(f"Success: {len(durations_boost)}/{n_eval}")

    nx.draw(g, pos, node_size=10, node_color='k')
    plt.plot([pos[v][0] for v in last_path_boost], [pos[v][1]
             for v in last_path_boost], 'g', alpha=.5)
    plt.plot([pos[v][0] for v in last_path_nx], [pos[v][1]
             for v in last_path_nx], 'r', alpha=.5)
    plt.show()
