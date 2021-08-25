import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from scipy.spatial import Delaunay

dtype = torch.float
device = torch.device("cpu")


def sample_points(n):
    points = np.random.random((n, 2))
    return torch.tensor(points, device=device,
                        dtype=dtype, requires_grad=True)


def make_graph(pos):
    N = pos.shape[0]
    g = nx.Graph()
    pos_np = pos.detach().numpy()

    tri = Delaunay(pos_np)
    (indptr, indices) = tri.vertex_neighbor_vertices
    for i in range(N):
        neighbors = indices[indptr[i]:indptr[i+1]]
        for n in neighbors:
            g.add_edge(i, n,
                       distance=np.linalg.norm(pos_np[i] - pos_np[n]))

    return g


def draw_graph(g, pos):
    plt.figure()
    N = pos.shape[0]
    pos_np = pos.detach().numpy()
    options = {
        "node_size": 20,
        "node_color": "red",
        "edgecolors": "black",
        "linewidths": 1,
        "width": 1,
        "with_labels": False
    }
    pos_dict = {i: pos_np[i] for i in range(N)}
    nx.draw_networkx(g, pos_dict, **options)


def make_paths(g, n_paths, seed=random.randint(0, 10E6)):
    np.random.seed(seed)
    N = nx.number_of_nodes(g)
    paths = []
    for _ in range(n_paths):
        start, goal = np.random.randint(0, N, (2))
        paths.append(
            nx.shortest_path(g, start, goal))
    return paths


def get_path_len(pos, path):
    sections = torch.zeros(len(path)-1)
    for i in range(len(path)-1):
        j = i+1
        sections[i] = torch.linalg.vector_norm(pos[i] - pos[j])
    return torch.sum(sections)


def get_paths_len(pos, paths):
    all_lens = torch.zeros(len(paths))
    for i, p in enumerate(paths):
        all_lens[i] = get_path_len(pos, p)
    return torch.sum(all_lens)


if __name__ == "__main__":
    n = 100
    batches = 2000
    learning_rate = 1e-4

    pos = sample_points(n)
    g = make_graph(pos)
    draw_graph(g, pos)

    test_paths = make_paths(g, 20, 0)
    test_costs = []
    for i_b in range(batches):
        test_length = get_paths_len(pos, test_paths)
        paths = make_paths(g, 10, i_b)
        length = get_paths_len(pos, paths)
        backward = length.backward(retain_graph=True)
        pos = torch.tensor(pos - pos.grad * learning_rate, requires_grad=True)
        g = make_graph(pos)
        if i_b % 10 == 0:
            print(f"test_length: {test_length}")
            print(f"length: {length}")
            test_costs.append(test_length.detach().numpy())

    draw_graph(g, pos)

    plt.figure()
    plt.plot(test_costs)
    plt.show()
