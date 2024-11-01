import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def plot_degree_dist(G):
    plt.figure(figsize=(8, 8))
    degrees = [G.degree(n) for n in G.nodes()]
    bin_borders = np.arange(min(degrees) - 0.5, max(degrees) + 1.5)
    plt.hist(degrees, bins=bin_borders)
    plt.xticks(
        bin_borders[:-1] + 0.5, map(lambda x: str(int(x)), bin_borders[:-1] + 0.5)
    )
    plt.grid()


def plot_graph(G):
    plt.figure(figsize=(8, 8))
    pos = {}
    for n in G.nodes:
        pos[n] = np.array(n)
    nx.draw_networkx_edges(G, pos, alpha=0.4)
    nx.draw_networkx_nodes(G, pos, node_size=80, cmap=plt.cm.Reds_r)
    plt.axis("off")


g = nx.grid_graph([3, 3])

g.remove_node((0, 0))
g.remove_node((0, 2))
g.remove_node((2, 0))
g.remove_node((2, 2))

plot_graph(g)
plot_degree_dist(g)
plt.show()
