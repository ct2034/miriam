import math
import pickle

import matplotlib.pyplot as plt
import networkx as nx

N = 300
G = nx.grid_graph((math.sqrt(N), math.sqrt(N)))
pos = nx.get_node_attributes(G, "pos")

plt.figure(figsize=(8, 8))
plt.title(str(G.number_of_nodes()))
nx.draw_networkx_edges(G, pos, alpha=0.4)
nx.draw_networkx_nodes(G, pos, node_size=50, node_color="#F00", cmap=plt.cm.Reds_r)


def filter1(x):
    return x < 200


view = nx.subgraph_view(G, filter1)

plt.figure(figsize=(8, 8))
plt.title(str(view.number_of_nodes()))
nx.draw_networkx_edges(view, pos, alpha=0.4)
nx.draw_networkx_nodes(view, pos, node_size=50, node_color="#0F0", cmap=plt.cm.Reds_r)


def filter2(x):
    return x < 100 | x >= 200


view = nx.subgraph_view(G, filter2)

plt.figure(figsize=(8, 8))
plt.title(str(view.number_of_nodes()))
nx.draw_networkx_edges(view, pos, alpha=0.4)
nx.draw_networkx_nodes(view, pos, node_size=50, node_color="#00F", cmap=plt.cm.Reds_r)

plt.show()

with open("test.pkl", "wb") as f:
    pickle.dump(view, f)
