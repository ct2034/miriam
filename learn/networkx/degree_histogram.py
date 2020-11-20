import matplotlib.pyplot as plt
import numpy as np

import networkx as nx


def plot_degree_dist(G):
    plt.figure(figsize=(8, 8))
    degrees = [G.degree(n) for n in G.nodes()]
    print(degrees)
    bin_borders = np.arange(min(degrees)-.5, max(degrees)+1.5)
    print(bin_borders)
    plt.hist(degrees, bins=bin_borders)
    plt.xticks(
        bin_borders[:-1] + .5,
        map(lambda x: str(int(x)), bin_borders[:-1] + .5)
    )
    plt.grid()


def plot_graph(G):
    plt.figure(figsize=(8, 8))
    pos = {}
    for n in G.nodes:
        pos[n] = np.array(n)
    nx.draw_networkx_edges(G, pos, alpha=0.4)
    nx.draw_networkx_nodes(G, pos,
                           node_size=80,
                           cmap=plt.cm.Reds_r)
    plt.axis('off')


g = nx.grid_graph([5, 5])

g.remove_node((1, 2))
g.remove_node((3, 3))
g.remove_node((4, 1))
g.remove_node((1, 0))

plot_graph(g)
plot_degree_dist(g)
plt.show()
