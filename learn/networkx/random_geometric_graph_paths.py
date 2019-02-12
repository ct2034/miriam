import networkx as nx
import random

def dist(a, b):
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


N = 1000

G = nx.random_geometric_graph(N, .1)

n = 100
for i in range(n):
    a = random.randint(0, N)
    b = random.randint(0, N)
    print(nx.astar_path(G, a, b))
