#!/usr/bin/env python3
from functools import reduce
from random import Random
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import png
import torch
from bresenham import bresenham
from libpysal import weights
from libpysal.cg import voronoi_frames
from networkx.exception import NetworkXNoPath, NodeNotFound
from pyflann import FLANN

from definitions import DISTANCE, MAP_IMG, PATH_W_COORDS, POS
from scenarios.visualization import get_colors
from tools import ProgressBar


def read_map(fname: str) -> MAP_IMG:
    """Read a map from a PNG file into a 2D tuple."""
    r = png.Reader(filename=fname)
    width, height, rows, _ = r.read()
    assert width == height
    data = []
    for i, row in enumerate(rows):
        skip = len(row) // width
        data.append(tuple(row[::skip]))
    return tuple(data)


def is_coord_free(
        map_img: MAP_IMG,
        point: Tuple[float, float]) -> bool:
    """Check if a coordinate is free."""
    size = len(map_img)
    try:
        return map_img[
            int(point[0] * size)
        ][
            int(point[1] * size)
        ] >= 255
    except IndexError:
        print(f"IndexError: {point}")


def is_pixel_free(
        map_img: MAP_IMG,
        point: Tuple[float, float]) -> bool:
    """Check if a pixel is free."""
    return map_img[
        int(point[0])
    ][
        int(point[1])
    ] >= 255


def check_edge(
        pos: Union[np.ndarray, torch.Tensor],
        map_img: MAP_IMG,
        a: int,
        b: int) -> bool:
    """Check edge between two nodes."""
    size = len(map_img)
    if a >= len(pos) or b >= len(pos):
        return False
    line = bresenham(
        int(pos[a][0]*size),
        int(pos[a][1]*size),
        int(pos[b][0]*size),
        int(pos[b][1]*size)
    )
    return all([is_pixel_free(map_img, (x[0], x[1])) for x in line])


def sample_points(
    n: int,
    map_img: MAP_IMG,
    rng: Random,
    free_points: bool = True
) -> torch.Tensor:
    """Sample `n` random points (0 <= x <= 1) from a map."""
    if free_points:
        test_fun = is_coord_free
    else:
        def test_fun(m, p): return not is_coord_free(m, p)
    points = np.empty(shape=(0, 2))
    while points.shape[0] < n:
        point = (rng.random(), rng.random())
        if test_fun(map_img, point):
            points = np.append(points, [np.array(point)], axis=0)
    return torch.tensor(points, device=torch.device("cpu"),
                        dtype=torch.float, requires_grad=True)


def make_graph_and_flann(
        pos: torch.Tensor,
        map_img: MAP_IMG,
        desired_n_nodes: int,
        rng: Random) -> Tuple[nx.Graph, FLANN]:
    """Convert array of node positions into graph by Delaunay Triangulation."""
    # add points in case they are not enough
    # while pos.shape[0] < desired_n_nodes:
    #     pos = torch.cat((pos, sample_points(1, map_img, rng)), dim=0)
    pos_np = pos.detach().numpy()
    # make dummy points in obstacles
    dummy_points = sample_points(
        len(pos_np) * .5, map_img, rng, free_points=False)
    pos_np_w_dummy = np.append(pos_np, dummy_points.detach().numpy(), axis=0)
    cells, _ = voronoi_frames(pos_np_w_dummy, clip="bbox")
    delaunay = weights.Rook.from_dataframe(cells, use_index=False)
    g: nx.Graph = delaunay.to_networkx()  # type: ignore
    nx.set_node_attributes(g, {
        i: pos_np[i] for i in range(len(pos_np))}, POS)
    nx.set_edge_attributes(g, [], DISTANCE)
    for a, b in g.edges():  # type: ignore
        if not check_edge(pos, map_img, a, b):
            g.remove_edge(a, b)
        else:
            g.edges[(a, b)][DISTANCE] = float(
                np.linalg.norm(pos_np[a] - pos_np[b]))
    # add self-edges
    for node in g.nodes():
        if not g.has_edge(node, node):
            g.add_edge(node, node)
            g.edges[(node, node)][DISTANCE] = 0.
    # remove isolated nodes
    subgraphs = list(nx.connected_components(g))
    i_main_graph = np.argmax([len(x) for x in subgraphs])
    g = g.subgraph(subgraphs[i_main_graph]).copy()
    remapping = {x: i for i, x in enumerate(g.nodes())}
    g = nx.relabel_nodes(g, remapping)
    pos_new = list(nx.get_node_attributes(g, POS).values())
    flann = FLANN(random_seed=0)
    flann.build_index(np.array(pos_new), random_seed=0)
    return g, flann


def draw_graph(
        g: nx.Graph,
        map_img: MAP_IMG = tuple(),
        paths: List[PATH_W_COORDS] = [],
        title: str = ""):
    """Display the graph and (optionally) paths."""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    g_cpy = nx.subgraph_view(g, filter_edge=lambda x, y: x != y)

    # Map Image
    if map_img:
        ax.imshow(
            np.swapaxes(np.array(map_img), 0, 1),
            cmap="gray",
            # alpha=.5,
            extent=(0, 1, 0, 1),
            origin="lower"
        )

    # Graph
    pos = nx.get_node_attributes(g_cpy, POS)
    pos_np = np.array([pos[i] for i in g_cpy.nodes()])
    options = {
        "ax": ax,
        "node_size": 20,
        "node_color": "black",
        "edgecolors": "grey",
        "linewidths": 1,
        "width": 1,
        "with_labels": False
    }
    pos_dict = {i: pos[i] for i in g_cpy.nodes()}
    nx.draw_networkx(g_cpy, pos_dict, **options)

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
        return nx.shortest_path(g, start, goal, DISTANCE)  # type: ignore
    except (NetworkXNoPath, nx.NodeNotFound):
        return None


def plan_path_between_coordinates(
        g: nx.Graph,
        flann: FLANN,
        start: Tuple[float, float],
        goal: Tuple[float, float]) -> Optional[PATH_W_COORDS]:
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
        map_img: MAP_IMG,
        rng: Random) -> List[PATH_W_COORDS]:
    """Make `n_paths` path between random coordinates on `g`."""
    pos_np = np.zeros((max(g.nodes) + 1, 2))
    for n, (x, y) in nx.get_node_attributes(g, POS).items():
        pos_np[n] = torch.tensor([x, y])
    flann = FLANN(random_seed=0)
    flann.build_index(np.array(pos_np, dtype=np.float32), random_seed=0)
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
        path: PATH_W_COORDS,
        training: bool) -> torch.Tensor:
    """Get the length of a path. With the sections from start coordinates to
    first node and from last node to goal coordinates beeing weighted
    `TAIL_WEIGHT` times more (if training is true, else once)."""
    TAIL_WEIGHT = 4
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
        paths: List[PATH_W_COORDS],
        training: bool) -> torch.Tensor:
    """Get the lengths of all paths in `paths`.
    If `training` is `True`, the tails are weighted more."""
    all_lens = torch.zeros(len(paths))
    for i, p in enumerate(paths):
        try:
            all_lens[i] = get_path_len(pos, p, training)
        except (NetworkXNoPath, NodeNotFound):
            pass
    return torch.sum(all_lens)


def optimize_poses(
        g: nx.Graph,
        pos: torch.Tensor,
        map_img: MAP_IMG,
        optimizer: torch.optim.Optimizer,
        n_nodes: int,
        rng: Random):
    """Optimize the poses of the nodes on `g` using `optimizer`."""
    test_paths = make_paths(g, 20, map_img, Random(0))
    test_length = get_paths_len(pos, test_paths, False)
    training_paths = make_paths(g, 10, map_img, rng)
    training_length = get_paths_len(pos, training_paths, True)
    _ = training_length.backward()
    optimizer.step()
    g, flann = make_graph_and_flann(pos, map_img, n_nodes, rng)
    optimizer.zero_grad()
    return g, pos, test_length, training_length


def optimize_poses_from_paths(
        g: nx.Graph,
        pos: torch.Tensor,
        path_set: List[List[PATH_W_COORDS]],
        map_img: MAP_IMG,
        n_nodes: int,
        optimizer: torch.optim.Optimizer,
        rng: Random):
    paths: List[PATH_W_COORDS]
    paths = reduce(lambda x, y: x + y, path_set, [])
    training_length = get_paths_len(pos, paths, True)
    if training_length.item() > 0.:
        _ = training_length.backward()
        optimizer.step()
        optimizer.zero_grad()
    g, flann = make_graph_and_flann(pos, map_img, n_nodes, rng)
    if len(paths) != 0:
        avg_len = training_length.item() / len(paths)
    else:
        avg_len = 0.
    return g, pos, flann, avg_len


if __name__ == "__main__":
    n = 64
    epochs = 2**14
    learning_rate = 1e-5
    stats_every = int(epochs / 64)
    map_fname: str = "roadmaps/odrm/odrm_eval/maps/x.png"
    rng = Random(0)

    map_img = read_map(map_fname)
    pos = sample_points(n, map_img, rng)
    g, flann = make_graph_and_flann(pos, map_img, n, rng)

    optimizer = torch.optim.Adam([pos], lr=learning_rate)
    test_costs = []

    vis_paths = make_paths(g, 3, map_img, Random(0))
    draw_graph(g, map_img, vis_paths)
    plt.savefig("roadmaps/var_odrm_torch/pre_training.png")
    pb = ProgressBar("Training", epochs)
    for i_e in range(epochs):
        g, pos, test_length, training_length = optimize_poses(
            g, pos, map_img, optimizer, n, rng)
        if i_e % stats_every == stats_every-1:
            print(f"test_length: {test_length}")
            print(f"training_length: {training_length}")
            test_costs.append(test_length.detach().numpy())
            pb.progress(i_e)
    pb.end()

    vis_paths = make_paths(g, 3, map_img, Random(0))
    draw_graph(g, map_img, vis_paths)
    plt.savefig("roadmaps/var_odrm_torch/post_training.png")

    plt.figure()
    plt.plot(test_costs)
    plt.savefig("roadmaps/var_odrm_torch/test_length.png")
    plt.show()
