#!/usr/bin/env python3
"""
compare: 
* https://en.wikipedia.org/wiki/Spectral_clustering
* https://en.wikipedia.org/wiki/Spectral_graph_theory

Create an G{n,m} random graph and compute the eigenvalues.
Requires numpy and matplotlib.
"""
import networkx as nx
import numpy.linalg
import matplotlib.pyplot as plt

n = 50  # nodes
m = 70  # edges
G = nx.gnm_random_graph(n, m)

L = nx.normalized_laplacian_matrix(G)
e = numpy.linalg.eigvals(L.A)
print("Largest eigenvalue:", max(e))
print("Smallest eigenvalue:", min(e))
plt.hist(e, bins=100)  # histogram with 100 bins
plt.xlim(0, 2)  # eigenvalues between 0 and 2

plt.figure()
pos = nx.spring_layout(G)
nx.draw(G, pos)

plt.show()
