#!/usr/bin/env python3
from functools import lru_cache
from random import Random
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import png
import torch
from bresenham import bresenham
from definitions import DISTANCE, POS
from libpysal import weights
from libpysal.cg import voronoi_frames
from networkx.exception import NetworkXNoPath, NodeNotFound
from pyflann import FLANN
from scenarios.visualization import get_colors
from tools import ProgressBar

dtype = torch.float
device = torch.device("cpu")
PATH_TYPE = Tuple[Tuple[float, float], Tuple[float, float], List[int]]


def read_map(fname: str) -> Tuple[Tuple[int, ...], ...]:
    """Read a map from a PNG file into a 2D tuple."""
    r = png.Reader(filename=fname)
    width, height, rows, _ = r.read()
    assert width == height
    data = []
    for i, row in enumerate(rows):
        data.append(tuple(row[::4]))
    return tuple(data)


# @lru_cache  # this slowed things down a lot. TODO: investigate
def is_coord_free(
        map_img: Tuple[Tuple[int, ...], ...],
        point: Tuple[float, float]) -> bool:
    """Check if a coordinate is free."""
    size = len(map_img)
    return map_img[
        int(point[0] * size)
    ][
        int(point[1] * size)
    ] >= 255


# @lru_cache  # this slowed things down a lot. TODO: investigate
def is_pixel_free(
        map_img: Tuple[Tuple[int, ...], ...],
        point: Tuple[float, float]) -> bool:
    """Check if a pixel is free."""
    return map_img[
        int(point[0])
    ][
        int(point[1])
    ] >= 255


def check_edge(
        pos: np.ndarray,
        map_img: Tuple[Tuple[int, ...], ...],
        a: int,
        b: int) -> bool:
    """Check edge between two nodes."""
    size = len(map_img)
    line = bresenham(
        int(pos[a][0]*size),
        int(pos[a][1]*size),
        int(pos[b][0]*size),
        int(pos[b][1]*size)
    )
    return all([is_pixel_free(map_img, (x[0], x[1])) for x in line])


def sample_points(
        n: int,
        map_img: Tuple[Tuple[int, ...], ...],
        rng: Random) -> torch.Tensor:
    """Sample `n` random points (0 <= x <= 1) from a map."""
    points = np.empty(shape=(0, 2))
    while points.shape[0] < n:
        point = (rng.random(), rng.random())
        if is_coord_free(map_img, point):
            points = np.append(points, [np.array(point)], axis=0)
    return torch.tensor(points, device=device,
                        dtype=dtype, requires_grad=True)


def make_graph(
        pos: torch.Tensor,
        map_img: Tuple[Tuple[int, ...], ...]) -> nx.Graph:
    """Convert array of node positions into graph by Delaunay Triangulation."""
    pos_np = pos.detach().numpy()
    cells, _ = voronoi_frames(pos_np, clip="bbox")
    delaunay = weights.Rook.from_dataframe(cells)
    g = delaunay.to_networkx()
    nx.set_node_attributes(g, {
        i: pos_np[i] for i in range(len(pos_np))}, POS)
    nx.set_edge_attributes(g, [], DISTANCE)
    for a, b in g.edges():
        if not check_edge(pos, map_img, a, b):
            g.remove_edge(a, b)
        else:
            g.edges[(a, b)][DISTANCE] = np.linalg.norm(pos_np[a] - pos_np[b])
    return g


def draw_graph(
        g: nx.Graph,
        map_img: Tuple[Tuple[int, ...], ...] = tuple(),
        paths: List[PATH_TYPE] = [],
        title: str = ""):
    """Display the graph and (optionally) paths."""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)

    # Map Image
    if map_img:
        ax.imshow(
            np.swapaxes(map_img, 0, 1),
            cmap="gray",
            alpha=.5,
            extent=(0, 1, 0, 1),
            origin="lower"
        )

    # Graph
    N = g.number_of_nodes()
    pos = nx.get_node_attributes(g, POS)
    pos_np = np.array([pos[i] for i in range(N)])
    options = {
        "ax": ax,
        "node_size": 20,
        "node_color": "black",
        "edgecolors": "grey",
        "linewidths": 1,
        "width": 1,
        "with_labels": False
    }
    pos_dict = {i: pos_np[i] for i in range(N)}
    nx.draw_networkx(g, pos_dict, **options)

    # Paths
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


def plan_path_between_nodes(
        g: nx.Graph,
        start: int,
        goal: int) -> Optional[List[int]]:
    """Plan a path on `g` between two nodes."""
    try:
        return nx.shortest_path(g, start, goal, DISTANCE)
    except (NetworkXNoPath, nx.NodeNotFound):
        return None


def plan_path_between_coordinates(
        g: nx.Graph,
        flann: FLANN,
        start: Tuple[float, float],
        goal: Tuple[float, float]) -> Optional[PATH_TYPE]:
    """Plan a path on `g` between two coordinates.
    Uses a FLANN first to find the nearest nodes to `start` and `goal`."""
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


def make_paths(
        g: nx.Graph,
        n_paths: int,
        map_img: Tuple[Tuple[int, ...], ...],
        rng: Random) -> List[PATH_TYPE]:
    """Make `n_paths` path between random coordinates on `g`."""
    pos = nx.get_node_attributes(g, POS)
    pos_np = np.array([pos[n] for n in g.nodes()])
    flann = FLANN()
    flann.build_index(np.array(pos_np))
    paths = []
    for i in range(n_paths):
        start = (rng.random(), rng.random())
        while not is_coord_free(map_img, start):
            start = (rng.random(), rng.random())
        goal = (rng.random(), rng.random())
        while not is_coord_free(map_img, goal):
            goal = (rng.random(), rng.random())
        path = plan_path_between_coordinates(g, flann, start, goal)
        if path is not None:
            paths.append(path)
    return paths


def get_path_len(
        pos: torch.Tensor,
        path: PATH_TYPE,
        training: bool) -> torch.Tensor:
    """Get the length of a path. With the sections from start coordinates to
    first node and from last node to goal coordinates beeing weighted 
    `TAIL_WEIGHT` times more (if training is true, else once)."""
    TAIL_WEIGHT = 3
    (start, goal, path_vs) = path
    sections = torch.zeros(len(path_vs)+1)
    for i in range(len(path_vs)-1):
        j = i+1
        sections[i] = torch.linalg.vector_norm(
            pos[path_vs[i]] - pos[path_vs[j]])
    s_tail = torch.linalg.vector_norm(
        torch.tensor(start) - pos[path_vs[0]])
    if training:
        sections[-2] = TAIL_WEIGHT * (s_tail ** 2 + s_tail)
    else:
        sections[-2] = s_tail
    g_tail = torch.linalg.vector_norm(
        torch.tensor(goal) - pos[path_vs[-1]])
    if training:
        sections[-1] = TAIL_WEIGHT * (g_tail ** 2 + g_tail)
    else:
        sections[-1] = g_tail
    return torch.sum(sections)


def get_paths_len(
        pos: torch.Tensor,
        paths: List[PATH_TYPE],
        training: bool) -> torch.Tensor:
    """Get the lengths of all paths in `paths`.
    If `training` is `True`, the tails are weighted more."""
    all_lens = torch.zeros(len(paths))
    for i, p in enumerate(paths):
        try:
            all_lens[i] = get_path_len(pos, p,
                                       training)
        except (NetworkXNoPath, NodeNotFound):
            pass
    return torch.sum(all_lens)


def optimize_poses(
        g: nx.Graph,
        pos: torch.Tensor,
        map_img: Tuple[Tuple[int, ...], ...],
        optimizer: torch.optim.Optimizer,
        rng: Random):
    """Optimize the poses of the nodes on `g` using `optimizer`."""
    test_paths = make_paths(g, 20, map_img, Random(0))
    test_length = get_paths_len(pos, test_paths, False)
    training_paths = make_paths(g, 10, map_img, rng)
    training_length = get_paths_len(pos, training_paths, True)
    _ = training_length.backward()
    optimizer.step()
    g = make_graph(pos, map_img)
    optimizer.zero_grad()
    return g, pos, test_length, training_length


if __name__ == "__main__":
    n = 100
    epochs = 300
    learning_rate = 1e-4
    stats_every = int(epochs / 50)
    map_fname: str = "roadmaps/odrm/odrm_eval/maps/z.png"
    rng = Random(0)

    map_img = read_map(map_fname)
    pos = sample_points(n, map_img, rng)
    g = make_graph(pos, map_img)
    test_paths = make_paths(g, 20, map_img, rng)

    optimizer = torch.optim.Adam([pos], lr=learning_rate)
    test_costs = []

    draw_graph(g, map_img, test_paths[:4])
    plt.savefig("roadmaps/var_odrm_torch/pre_training.png")
    pb = ProgressBar("Training", epochs)
    for i_e in range(epochs):
        g, pos, test_length, training_length = optimize_poses(
            g, pos, map_img, optimizer, rng)
        if i_e % stats_every == stats_every-1:
            print(f"test_length: {test_length}")
            print(f"training_length: {training_length}")
            test_costs.append(test_length.detach().numpy())
            pb.progress(i_e)
    pb.end()

    draw_graph(g, map_img, test_paths[:4])
    plt.savefig("roadmaps/var_odrm_torch/post_training.png")

    plt.figure()
    plt.plot(test_costs)
    plt.savefig("roadmaps/var_odrm_torch/test_length.png")
    plt.show()
