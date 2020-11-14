import matplotlib.pyplot as plt
import numpy as np

import networkx as nx


def plot_degree_dist(G):
    plt.figure(figsize=(8, 8))
    degrees = [G.degree(n) for n in G.nodes()]
    plt.hist(degrees, range=[min(degrees), max(degrees)], align='mid')


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


g = nx.grid_graph([8, 8])

g.remove_node((1, 2))
g.remove_node((3, 3))
g.remove_node((4, 1))
g.remove_node((7, 3))
g.remove_node((6, 4))
g.remove_node((1, 5))

plot_graph(g)
plot_degree_dist(g)
plt.show()
