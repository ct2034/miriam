from math import sqrt

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

import networkx as nx
from pyflann import *


def dist(a, b):
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


N = 300

G = nx.random_geometric_graph(N, 0.1)

# position is stored as node attribute data for random_geometric_graph
pos = nx.get_node_attributes(G, 'pos')
pos2 = pos.copy()
pos3 = pos.copy()

plt.figure(figsize=(8, 8))
plt.title("r-disc")
nx.draw_networkx_edges(G, pos, alpha=0.4)
nx.draw_networkx_nodes(G, pos,
                       node_size=50,
                       node_color='#0F1C95',
                       cmap=plt.cm.Reds_r)

plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.axis('off')

# vgl nearest neighbor graph

points = np.array(list(pos.values()))
G2 = nx.Graph()
G2.add_nodes_from(range(N))

nn = 4
flann = FLANN()
result, dists = flann.nn(
    points, points, nn,
    algorithm="kmeans", branching=32, iterations=7, checks=16)

for i in range(N):
    for inn in range(nn):
        G2.add_edge(i, result[i, inn])

plt.figure(figsize=(8, 8))
plt.title("K-Nearest-Neighbour")
nx.draw_networkx_edges(G2, pos2, alpha=0.4)
nx.draw_networkx_nodes(G2, pos2,
                       node_size=50,
                       node_color='#0F1C95',
                       cmap=plt.cm.Reds_r)

plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.axis('off')


# vgl online nearest neighbor graph


points = list(pos3.values())
G3 = nx.Graph()
G3.add_nodes_from(range(N))

for i in range(N):
    for j in range(N):
        d_ij = dist(points[i], points[j])
        d_min = 10.
        min_k = 0
        for k in range(j):
            d = dist(points[k], points[j])
            if d < d_min:
                d_min = d
                min_k = k
        if d_min == d_ij:
            G3.add_edge(i, j)

plt.figure(figsize=(8, 8))
plt.title("Online Nearest-Neighbour")
nx.draw_networkx_edges(G3, pos3, alpha=0.4)
nx.draw_networkx_nodes(G3, pos3,
                       node_size=50,
                       node_color='#0F1C95',
                       cmap=plt.cm.Reds_r)

plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.axis('off')


# vgl delauny ---
# ---------------
points = np.array(list(pos.values()))
tri = Delaunay(points)

plt.figure(figsize=(8, 8))
plt.title("Delaunay")
plt.triplot(points[:, 0], points[:, 1], tri.simplices.copy(), '-k', alpha=0.4)
plt.plot(points[:, 0], points[:, 1], 'ob')

plt.axis('off')
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.show()
