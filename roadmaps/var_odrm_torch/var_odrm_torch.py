import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from networkx.exception import NetworkXNoPath, NodeNotFound
from pyflann import FLANN
from scenarios.visualization import get_colors
from scipy.spatial import Delaunay
from tools import ProgressBar

dtype = torch.float
device = torch.device("cpu")


def sample_points(n):
    points = np.random.random((n, 2))
    return torch.tensor(points, device=device,
                        dtype=dtype, requires_grad=True)


def make_graph(pos):
    """Convert array of node positions into graph by Delaunay Triangulation."""
    N = pos.shape[0]
    g = nx.Graph()
    pos_np = pos.detach().numpy()
    # to discourage too many edges parallel to walls,
    # these will be ignored after Delaunay because they are not added by any
    # edge below.
    n_fn = 5
    fake_nodes = np.array(
        [(0, 1/n_fn*i) for i in range(n_fn+1)] +
        [(1, 1/n_fn*i) for i in range(n_fn+1)] +
        [(1/n_fn*i, 0) for i in range(n_fn+1)] +
        [(1/n_fn*i, 1) for i in range(n_fn+1)]
    )
    pos_np = np.append(pos_np, fake_nodes, axis=0)
    tri = Delaunay(pos_np)
    (indptr, indices) = tri.vertex_neighbor_vertices
    for i in range(N):
        neighbors = indices[indptr[i]:indptr[i+1]]
        for n in neighbors:
            if n >= N:
                continue  # ignoring fake nodes form above
            g.add_edge(i, n,
                       distance=np.linalg.norm(pos_np[i] - pos_np[n]))
    return g


def draw_graph(g, pos, paths=[]):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    N = pos.shape[0]
    pos_np = pos.detach().numpy()
    options = {
        "ax": ax,
        "node_size": 20,
        "node_color": "red",
        "edgecolors": "black",
        "linewidths": 1,
        "width": 1,
        "with_labels": False
    }
    pos_dict = {i: pos_np[i] for i in range(N)}
    nx.draw_networkx(g, pos_dict, **options)

    colors = get_colors(len(paths))
    style = '--'
    for i_p, p in enumerate(paths):
        (start, goal, path_vs) = p
        for i in range(len(path_vs)-1):
            j = i+1
            ax.plot(
                [pos_np[path_vs[i], 0], pos_np[path_vs[j], 0]],  # x
                [pos_np[path_vs[i], 1], pos_np[path_vs[j], 1]],  # y
                style,
                color=colors[i_p])
        ax.plot(
            [start[0], pos_np[path_vs[0], 0]],  # x
            [start[1], pos_np[path_vs[0], 1]],  # y
            style,
            color=colors[i_p])
        ax.plot(
            [goal[0], pos_np[path_vs[-1], 0]],  # x
            [goal[1], pos_np[path_vs[-1], 1]],  # y
            style,
            color=colors[i_p])
        ax.set_xlim([0., 1.])
        ax.set_ylim([0., 1.])


def plan_path_between_nodes(g, start, goal):
    try:
        return nx.shortest_path(g, start, goal, 'distance')
    except (NetworkXNoPath, nx.NodeNotFound):
        return None


def plan_path_between_coordinates(g, flann, start, goal):
    nn = 1
    result, _ = flann.nn_index(
        np.array([start, goal], dtype="float32"), nn,
        random_seed=0)
    start_v, goal_v = result
    node_path = plan_path_between_nodes(g, start_v, goal_v)
    if node_path is not None:
        return (start, goal, node_path)
    else:
        return None


def make_paths(g, pos, n_paths, seed=random.randint(0, 10E6)):
    pos_np = pos.detach().numpy()
    flann = FLANN()
    flann.build_index(np.array(pos_np))
    paths = []
    for i in range(n_paths):
        np.random.seed(seed+i)
        [start, goal] = np.random.random((2, 2))
        path = plan_path_between_coordinates(g, flann, start, goal)
        if path is not None:
            paths.append(path)
    return paths


def get_path_len(pos, path):
    TAIL_WEIGHT = 3
    (start, goal, path_vs) = path
    sections = torch.zeros(len(path_vs)+1)
    for i in range(len(path_vs)-1):
        j = i+1
        sections[i] = torch.linalg.vector_norm(
            pos[path_vs[i]] - pos[path_vs[j]])
    s_tail = torch.linalg.vector_norm(
        torch.tensor(start) - pos[path_vs[0]])
    sections[-2] = TAIL_WEIGHT * (s_tail ** 2 + s_tail)
    g_tail = torch.linalg.vector_norm(
        torch.tensor(goal) - pos[path_vs[-1]])
    sections[-1] = TAIL_WEIGHT * (g_tail ** 2 + g_tail)
    return torch.sum(sections)


def get_paths_len(pos, paths):
    all_lens = torch.zeros(len(paths))
    for i, p in enumerate(paths):
        try:
            all_lens[i] = get_path_len(pos, p)
        except (NetworkXNoPath, NodeNotFound):
            pass
    return torch.sum(all_lens)


def optimize_poses(g, pose, optimizer):
    test_paths = make_paths(g, pos, 20, 0)
    test_length = get_paths_len(pos, test_paths)
    training_paths = make_paths(g, pos, 10, i_e)
    training_length = get_paths_len(pos, training_paths)
    backward = training_length.backward()
    optimizer.step()
    g = make_graph(pos)
    optimizer.zero_grad()
    return g, pose, test_length, training_length


if __name__ == "__main__":
    n = 100
    epochs = 300
    learning_rate = 1e-4
    stats_every = int(epochs / 50)

    pos = sample_points(n)
    g = make_graph(pos)
    test_paths = make_paths(g, pos, 20, 0)

    optimizer = torch.optim.Adam([pos], lr=learning_rate)
    test_costs = []

    draw_graph(g, pos, test_paths[:4])
    plt.savefig("pre_training.png")
    pb = ProgressBar("Training", epochs)
    for i_e in range(epochs):
        g, pos, test_length, training_length = optimize_poses(
            g, pos, optimizer)
        if i_e % stats_every == stats_every-1:
            print(f"test_length: {test_length}")
            print(f"training_length: {training_length}")
            test_costs.append(test_length.detach().numpy())
            pb.progress(i_e)
    pb.end()

    draw_graph(g, pos, test_paths[:4])
    plt.savefig("post_training.png")

    plt.figure()
    plt.plot(test_costs)
    plt.savefig("test_length.png")
    plt.show()
