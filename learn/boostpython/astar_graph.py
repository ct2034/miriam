#!/usr/bin/env python3

import random

import networkx as nx
from libastar_graph import AstarSolver

if __name__ == "__main__":
    g = nx.random_geometric_graph(100, 0.15)
    pos = nx.get_node_attributes(g, 'pos')
    posl = [[pos[v][0], pos[v][1]] for v in g.nodes()]
    edges = [[e[0], e[1]] for e in g.edges()]
    a = AstarSolver(posl, edges)
    print(a)
    print(a.retreive(random.randrange(g.number_of_nodes())))
    print(a.retreive(random.randrange(g.number_of_nodes())))
    print(a.retreive(random.randrange(g.number_of_nodes())))
    print("#"*20)
    print(a.plan(random.randrange(g.number_of_nodes()),
          random.randrange(g.number_of_nodes())))
    print(a.plan(random.randrange(g.number_of_nodes()),
          random.randrange(g.number_of_nodes())))
    print(a.plan(random.randrange(g.number_of_nodes()),
          random.randrange(g.number_of_nodes())))
