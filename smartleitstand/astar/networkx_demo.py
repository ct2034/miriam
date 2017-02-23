import timeit

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from smartleitstand.astar import astar_grid48con

eight_con = False
n = 10
G = nx.grid_graph([n, n])
map = np.zeros([n, n])
map[:n * .8, n * .2] = -1
map[n * .8, n * .2:n * .6] = -1
map[n * .2, n * .4:n * .8] = -1
map[n * .2:, n * .8] = -1

start = (n * .1, n * .1)
goal = (n * .9, n * .9)


def cost(a, b):
    if map[a] >= 0 and map[b] >= 0:  # no obstacle
        return astar_grid48con.cost(a, b, map)
    else:
        return np.Inf


obstacle = []
for n in G.nodes_iter():
    if not map[n] >= 0:  # obstacle
        obstacle.append(n)

G.remove_nodes_from(obstacle)

if eight_con:
    # add 8-connectedness
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

print("length: ", astar_grid48con.path_length(path))

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

n = G.number_of_nodes()
if n < 500:
    plt.figure()
    pos = nx.spring_layout(G, iterations=1000, k=.1 / np.sqrt(n))
    nx.draw(G, pos)

plt.show()
