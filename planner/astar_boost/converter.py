import networkx as nx

from planner.astar_boost.build.libastar_graph import AstarSolver


def initialize_from_graph(g):
    pos = nx.get_node_attributes(g, "pos")
    posl = [[0.0, 0.0]] * (max(pos.keys()) + 1)
    for k, v in pos.items():
        posl[k] = [float(v[0]), float(v[1])]
    edges = [[int(e[0]), int(e[1])] for e in g.edges()]
    a = AstarSolver(posl, edges)
    return a
