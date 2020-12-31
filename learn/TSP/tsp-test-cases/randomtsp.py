#!/usr/bin/env python

import sys
import random
import math
import networkx as nx


def euclidian(G, i, j):
    return math.sqrt(
        (G.node[i]['x'] - G.node[j]['x']) ** 2 +
        (G.node[i]['y'] - G.node[j]['y']) ** 2
    )


def main():
    n = int(sys.argv[1])
    out = sys.argv[2]
    ans = float(sys.argv[3]) if len(sys.argv) >= 4 else None
    G = nx.Graph()

    # create nodes randomly placed in [0,100)x[0,100)
    P = [None] * n
    for i in xrange(n):
        G.add_node(i, x=random.random() * 100, y=random.random() * 100)

    # create a complete weighted graph using Euclidian distances
    for i in xrange(n):
        for j in xrange(i + 1, n):
            G.add_edge(i, j, weight=euclidian(G, i, j))

    # embed a tour with edge weight = ans (usually 0)
    if ans is not None:
        T = list(G.nodes())
        random.shuffle(T)
        for i in xrange(1, len(T)):
            G.edge[T[i - 1]][T[i]]['weight'] = ans
        G.edge[T[-1]][T[0]]['weight'] = ans
        print
        T

    nx.write_gexf(G, out)
    print
    "n=%d m=%d" % (G.number_of_nodes(), G.number_of_edges())


if __name__ == "__main__":
    main()
