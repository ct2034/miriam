
from bresenham import bresenham
from itertools import product
import math
import matplotlib.pyplot as plt
from multiprocessing import Pool
import networkx as nx
import numpy as np
from scipy.spatial import Delaunay
from pyflann import FLANN

MAX_COST = 100000
pool = Pool()


def is_pixel_free(im, p):
    return min(im[
        int(p[1]),
        int(p[0])
    ]) > 250


def get_random_pos(im):
    im_shape = im.shape
    p = np.random.rand(2) * im_shape[0]
    while (not is_pixel_free(im, p)):
        p = np.random.rand(2) * im_shape[0]
    return p


def dist(sx, sy, gx, gy):
    return math.sqrt((sx-gx)**2+(sy-gy)**2)


def init_graph_posar(im, N):
    return np.array([get_random_pos(im) for _ in range(N)])


def graph_from_posar(N, posar):
    g = nx.Graph()
    g.add_nodes_from(range(N))
    pos = nx.get_node_attributes(g, 'pos')
    for i in range(N):
        pos[i] = posar[i]
    return g, pos


def make_edges(N, g, posar, im):
    tri = Delaunay(posar)
    (indptr, indices) = tri.vertex_neighbor_vertices
    for i in range(N):
        neigbours = indices[indptr[i]:indptr[i+1]]
        for n in neigbours:
            if i < n:
                line = bresenham(
                    int(posar[i][0]),
                    int(posar[i][1]),
                    int(posar[n][0]),
                    int(posar[n][1])
                )
                # print(list(line))
                if all([is_pixel_free(im, x) for x in line]):
                    g.add_edge(i, n, distance=dist(
                        sx=posar[i][0], sy=posar[i][1],
                        gx=posar[n][0], gy=posar[n][1]
                    ))


def plot_graph(fig, ax, g, pos, im, fname=''):
    nx.draw_networkx_nodes(g, pos, ax=ax, node_size=20)
    nx.draw_networkx_edges(g, pos, ax=ax, width=0.5, alpha=0.6)
    ax.imshow(im)
    ax.axis('off')
    fig.add_axes(ax)
    if(fname):
        fig.savefig(fname)
    else:
        plt.show()
    plt.close('all')


def path(start, goal, nn, g, posar):
    flann = FLANN()
    result, dists = flann.nn(
        posar, np.array([start, goal]), nn,
        algorithm="kmeans", branching=32, iterations=7, checks=16)
    min_c = MAX_COST
    min_p = None
    for (i_s, i_g) in product(range(nn), range(nn)):
        try:
            c = nx.shortest_path_length(g,
                                        result[0][i_s],
                                        result[1][i_g],
                                        weight='distance'
                                        )
        except nx.exception.NetworkXNoPath:
            c = MAX_COST
        if c < min_c:
            min_p = nx.shortest_path(g,
                                     result[0][i_s],
                                     result[1][i_g],
                                     weight='distance'
                                     )
            min_c = c + dists[0][i_s] + dists[1][i_g]
    # assert min_c != MAX_COST, "no path"
    return min_c, min_p


def plot_path(fig, start, goal, path, posar):
    xs = [start[0]]
    ys = [start[1]]
    for v in path:
        xs.append(posar[v][0])
        ys.append(posar[v][1])
    xs.append(goal[0])
    ys.append(goal[1])
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.plot(xs, ys, 'b')
    return ax


def eval(t, evalset, nn, g, pos, posar, im):
    cost = 0
    unsuccesful = 0
    ne = evalset.shape[0]
    for i in range(ne):
        (c, p) = path(evalset[i, 0], evalset[i, 1], nn, g, posar)
        if c == MAX_COST:
            unsuccesful += 1
        else:
            cost += c
    if t > -1 & t % 10 == 0:
        fig = plt.figure(figsize=[8, 8])
        ax = plot_path(fig, evalset[ne-1, 0], evalset[ne-1, 1], p, posar)
        plot_graph(fig, ax, g, pos, im, fname="anim/frame%04d.png"%t)
    return cost / (ne-unsuccesful), unsuccesful


def grad_func(x, batch, nn, g, posar):
    out = np.zeros(shape=x.shape)
    for i_b in range(batch.shape[0]):
        (c, p) = path(batch[i_b, 0], batch[i_b, 1], nn, g, posar)
        if c != MAX_COST:
            coord_p = np.zeros([len(p) + 2, 2])
            coord_p[0, :] = batch[i_b, 0]
            coord_p[1:(1+len(p)), :] = np.array([x[i_p] for i_p in p])
            coord_p[(1+len(p)), :] = batch[i_b, 1]
            for i_p in range(len(p)):
                i_cp = i_p + 1
                for j in [0, 1]:
                    out[p[i_p], j] += (
                        (coord_p[i_cp, j] - coord_p[i_cp-1, j])
                        / math.sqrt((coord_p[i_cp, 0] - coord_p[i_cp-1, 0]
                                     )**2
                                    + (coord_p[i_cp, 1] - coord_p[i_cp-1, 1]
                                       )**2)
                        + (coord_p[i_cp, j] - coord_p[i_cp+1, j])
                        / math.sqrt((coord_p[i_cp, 0] - coord_p[i_cp+1, 0]
                                     )**2
                                    + (coord_p[i_cp, 1] - coord_p[i_cp+1, 1]
                                       )**2)
                    )
    return out


def fix(posar_prev, posar, im):
    for i in range(posar.shape[0]):
        if(not is_pixel_free(im, posar[i])):
            posar[i] = posar_prev[i]
