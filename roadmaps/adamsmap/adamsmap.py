
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.spatial import Delaunay
from pyflann import *

#Graph
N = 1000

#Training
ntb = 100 #number of batches
nts = 100 #batch size

#Evaluation
ne = 50 #evaluation set size

def get_random_pos():
    p = np.array([0.5, 0.5])
    while ((p[0] > 0.4) &
           (p[0] < 0.6) &
           (p[1] > 0.4) &
           (p[1] < 0.6)):
        p = np.random.rand(2)
    return p

g = nx.Graph()
g.add_nodes_from(range(N))
pos = nx.get_node_attributes(g, 'pos')
for i in range(N):
    pos[i] = get_random_pos()

tri = Delaunay([pos[i] for i in range(N)])
(indptr, indices) = tri.vertex_neighbor_vertices
for i in range(N):
    neigbours = indices[indptr[i]:indptr[i+1]]
    for n in neigbours:
        if i < n:
            g.add_edge(i, n)

plt.figure(figsize=(8, 8))
nx.draw_networkx_nodes(g, pos, node_size=30)
nx.draw_networkx_edges(g, pos, width=0.5, alpha=0.6)
plt.axis('off')
plt.show()

evalset = np.random.rand(ne, 2, 2)
nn = 1
flann = FLANN()
result, dists = flann.nn(
    points, points, nn,
    algorithm="kmeans", branching=32, iterations=7, checks=16)
