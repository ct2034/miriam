import timeit

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from astar import astar_grid8con

n = 100
G = nx.grid_graph([n, n])
map = np.zeros([100, 100])
map[:80, 20] = -1
map[80, 20:60] = -1
map[20, 40:80] = -1
map[20:, 80] = -1

start = (10, 10)
goal = (90, 90)

def cost(a, b):
    if map[a] >= 0 and map[b] >= 0:  # no obstacle
        return astar_grid8con.cost(a, b, map)
    else:
        return np.Inf


obstacle = []
for n in G.nodes_iter():
    if not map[n] >= 0:  # obstacle
        obstacle.append(n)

G.remove_nodes_from(obstacle)

for n in G.nodes_iter():
    for d in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
        checking = (d[0] + n[0], d[1] + n[1])
        if G.has_node(checking):
            if not G.has_edge(n, checking):
                G.add_edge(n, checking)
            G[n][checking]['weight'] = cost(n, checking)

t = timeit.Timer()
t.timeit()

path = nx.astar_path(G, start, goal, cost)

print("computation time:", t.repeat(), "s")

print("length: ", astar_grid8con.path_length(path))

fig, ax = plt.subplots()

ax.imshow(map.T, cmap='Greys', interpolation='nearest')
ax.set_title('astar path')
ax.axis([0, map.shape[0], 0, map.shape[1]])
ax.plot(
    np.array(np.matrix(path)[:, 0]).flatten(),
    np.array(np.matrix(path)[:, 1]).flatten(),
    c='b',
    lw=2
)

plt.show()
