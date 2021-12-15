#!/usr/bin/env python3
import time
from random import Random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import png
from bresenham import bresenham
from libpysal import weights
from libpysal.cg import voronoi_frames
from scipy.spatial import Delaunay as Delaunay
from tqdm import tqdm


class Delaunay_impl:
    def __init__(self, points: np.ndarray, n_fake_nodes: int):
        raise NotImplementedError

    def get_delaunay_graph(self) -> nx.Graph:
        raise NotImplementedError


class Delaunay_scipy(Delaunay_impl):
    def __init__(self, points: np.ndarray, n_fake_nodes: int):
        self.points = points
        self.n_fake_nodes = n_fake_nodes

    def get_delaunay_graph(self) -> nx.Graph:
        n_nodes = self.points.shape[0] - self.n_fake_nodes
        tri = Delaunay(self.points)
        (indptr, indices) = tri.vertex_neighbor_vertices
        g = nx.Graph()
        for i in range(n_nodes):
            neighbors = indices[indptr[i]:indptr[i+1]]
            for n in neighbors:
                if n >= n_nodes:
                    continue  # ignoring fake nodes from above
                if check_edge(self.points, map_img, i, n):
                    g.add_edge(i, n,
                               distance=np.linalg.norm(
                                   self.points[i] - self.points[n]))
        nx.set_node_attributes(g,
                               {i: tuple(self.points[i])
                                for i in range(n_nodes)},
                               'pos')
        return g


class Delaunay_libpysal(Delaunay_impl):
    def __init__(self, points: np.ndarray, n_fake_nodes: int):
        self.points = points
        self.n_fake_nodes = n_fake_nodes

    def get_delaunay_graph(self) -> nx.Graph:
        n_nodes = self.points.shape[0] - self.n_fake_nodes
        cells, generators = voronoi_frames(self.points, clip="convex hull")
        delaunay = weights.Rook.from_dataframe(cells)
        g_tmp = delaunay.to_networkx()
        g = nx.Graph()
        for e in g_tmp.edges():
            (a, b) = e
            if a >= n_nodes or b >= n_nodes:
                continue  # ignoring fake nodes from above
            if check_edge(self.points, map_img, a, b):
                g.add_edge(a, b,
                           distance=np.linalg.norm(
                               self.points[a] - self.points[b]))
        nx.set_node_attributes(g,
                               {i: tuple(self.points[i])
                                for i in range(n_nodes)},
                               'pos')
        return g


def read_map(fname: str):
    r = png.Reader(filename=fname)
    width, height, rows, info = r.read()
    assert width == height
    data = []
    for i, row in enumerate(rows):
        data.append(tuple(row[::4]))
    return np.array(data)


def is_pixel_free(map_img, point):
    return map_img[
        int(point[0])
    ][
        int(point[1])
    ] >= 255


def check_edge(pos, map_img, a, b):
    line = bresenham(
        pos[a][0],
        pos[a][1],
        pos[b][0],
        pos[b][1]
    )
    # print(list(line))
    return all([is_pixel_free(map_img, tuple(x)) for x in line])


def prepare_fake_nodes(pos: np.ndarray) -> int:
    """add fake nodes to pos to account for map borders"""
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
    pos = np.append(pos, fake_nodes, axis=0)
    return fake_nodes.shape[0]  # number of fake nodes


def run_delaunay_scipy(pos_np, map_img):
    n_fn = prepare_fake_nodes(pos_np)
    ds = Delaunay_scipy(pos_np, n_fn)
    g = ds.get_delaunay_graph()
    return g


def run_delaunay_libpysal(pos_np, map_img):
    n_fn = prepare_fake_nodes(pos_np)
    ds = Delaunay_libpysal(pos_np, n_fn)
    g = ds.get_delaunay_graph()
    return g


if __name__ == '__main__':
    n_nodes = 1000
    map_fname: str = "roadmaps/odrm_eval/maps/z.png"
    map_img = read_map(map_fname)
    assert map_img.shape[0] == map_img.shape[1], "map must be square"

    n_runs = 1000
    success_s_scipy = 0
    success_s_libpysal = 0
    times_scipy = []
    times_libpysal = []
    rng = Random(0)

    for _ in tqdm(range(n_runs)):
        pos = np.array(
            [(rng.randint(0, map_img.shape[0]-1),
              rng.randint(0, map_img.shape[1]-1)) for _ in range(n_nodes)],
            dtype=np.int32)

        # randomly change order
        p = rng.random()
        HALF = .5

        def eval_scipy():
            try:
                start_scipy = time.time()
                g_sp = run_delaunay_scipy(pos, map_img)
                times_scipy.append(time.time() - start_scipy)
                return g_sp, True
            except Exception as e:
                print(e)
                return None, False

        def eval_libpysal():
            try:
                start_libpysal = time.time()
                g_lp = run_delaunay_libpysal(pos, map_img)
                times_libpysal.append(time.time() - start_libpysal)
                return g_lp, True
            except Exception as e:
                print(e)
                return None, False

        if p < HALF:
            g_sp, success_scipy = eval_scipy()
        g_lp, success_lipysal = eval_libpysal()
        if p >= HALF:
            g_sp, success_scipy = eval_scipy()
        success_s_scipy += int(success_scipy)
        success_s_libpysal += int(success_lipysal)

    # show last results
    plt.figure()
    plt.imshow(np.swapaxes(map_img, 0, 1), cmap='gray')
    nx.draw_networkx(g_sp, pos=pos, node_size=10,
                     node_color='r', with_labels=False)
    plt.figure()
    plt.imshow(np.swapaxes(map_img, 0, 1), cmap='gray')
    nx.draw_networkx(g_lp, pos=pos, node_size=10,
                     node_color='r', with_labels=False)

    # print success rate
    print("success rate scipy:", success_s_scipy / n_runs)
    print("success rate libpysal:", success_s_libpysal / n_runs)
    # success rate scipy: 1.0
    # success rate libpysal: 0.988

    # show runtimes
    plt.figure()
    plt.violinplot([times_scipy, times_libpysal])
    plt.ylabel("runtime [s]")
    plt.xticks([1, 2], ['scipy', 'libpysal'])
    plt.savefig("learn/delaunay_benchmark/benchmark.png")
    # plt.show()
