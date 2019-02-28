#!python3
from bresenham import bresenham
from functools import reduce
from itertools import product
import math
from matplotlib import cm
import matplotlib.pyplot as plt
from multiprocessing import Pool
import networkx as nx
import numpy as np
from scipy.spatial import Delaunay
from pyflann import FLANN

MAX_COST = 100000
END_BOOST = 3.
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


def dist(a, b):
    return math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)


def dist_posar(an, bn):
    global posar
    return dist(posar[an], posar[bn])


def edge_cost_factor(a, b, edgew):
    def sigmoid(x):
        return 2. / (1. + math.exp(-x)) + 1
    if a < b:
        c = (edgew[a, b] - 1)**2 + 1
    else:  # b < a
        c = (1 - edgew[b, a])**2 + 1
    # print(c)
    return sigmoid(c)


def path_cost(p, posar, edgew, prin=False):
    if edgew is not None:  # ge
        return reduce(lambda x, y: x+y,
                      [dist(posar[p[i]], posar[p[i+1]])
                       * edge_cost_factor(p[i], p[i+1], edgew)
                       for i in range(len(p)-1)], 0.)
    else:
        return reduce(lambda x, y: x+y,
                      [dist(posar[p[i]], posar[p[i+1]])
                       for i in range(len(p)-1)], 0.)
    # TODO: use graph weight


def init_graph_posar_edgew(im, N):
    global posar
    posar = np.array([get_random_pos(im) for _ in range(N)])
    edgew = np.triu(np.random.normal(loc=0, scale=0.5, size=(N, N)), 1)
    return posar, edgew


def graphs_from_posar(N, _posar):
    global posar
    posar = _posar
    g = nx.DiGraph()
    g.add_nodes_from(range(N))
    ge = nx.DiGraph()
    ge.add_nodes_from(range(N))
    pos = nx.get_node_attributes(g, 'pos')
    for i in range(N):
        pos[i] = posar[i]
    return g, ge, pos


def make_edges(N, g, ge, posar, edgew, im):
    b = im.shape[0]
    fakenodes1 = np.array(np.array(list(
        product([0, b], np.linspace(0, b, 6)))))
    fakenodes2 = np.array(np.array(list(
        product(np.linspace(0, b, 6), [0, b]))))
    tri = Delaunay(np.append(posar, np.append(
        fakenodes1, fakenodes2, axis=0), axis=0
    ))
    (indptr, indices) = tri.vertex_neighbor_vertices
    for i in range(N):
        neigbours = indices[indptr[i]:indptr[i+1]]
        for n in neigbours:
            if i < n & n < N:
                line = bresenham(
                    int(posar[i][0]),
                    int(posar[i][1]),
                    int(posar[n][0]),
                    int(posar[n][1])
                )
                # print(list(line))
                if all([is_pixel_free(im, x) for x in line]):
                    g.add_edge(i, n,
                               distance=dist(posar[i], posar[n])
                               * edge_cost_factor(i, n, edgew))
                    g.add_edge(n, i,
                               distance=dist(posar[i], posar[n])
                               * edge_cost_factor(n, i, edgew))
                    if edgew[i, n] > 0:
                        ge.add_edge(i, n,
                                    distance=dist(posar[i], posar[n]))
                    else:
                        ge.add_edge(n, i,
                                    distance=dist(posar[i], posar[n]))


def plot_graph(fig, ax, g, pos, edgew, im, fname=''):
    nx.draw_networkx_nodes(g, pos, ax=ax, node_size=15, node_color='k')

    def show_edge(e):
        if e[0] < e[1]:
            return edgew[e[0], e[1]] > 0
        else:  # e[1] < e[0]
            return edgew[e[1], e[0]] < 0

    edges = list(filter(show_edge, g.edges()))
    edge_colors = [cm.brg(.5 * abs(val) + .5) for val in
                   map(lambda x: edgew[x[0], x[1]] if x[0] < x[1]
                       else edgew[x[1], x[0]], edges)]
    nx.draw_networkx_edges(g, pos, ax=ax, edgelist=edges,
                           width=0.7, edge_color=edge_colors)
    ax.imshow(im)
    ax.axis('off')
    fig.add_axes(ax)
    if(fname):
        fig.savefig(fname)
    else:
        plt.show()
    plt.close('all')


def path(start, goal, nn, g, posar, edgew):
    flann = FLANN()
    result, dists = flann.nn(
        posar, np.array([start, goal]), nn,
        algorithm="kmeans", branching=32, iterations=7, checks=16)
    min_c = MAX_COST
    min_p = None
    for (i_s, i_g) in product(range(nn), range(nn)):
        try:
            p = nx.astar_path(g,
                              result[0][i_s],
                              result[1][i_g],
                              heuristic=dist_posar,
                              weight='distance'
                              )
            c = (path_cost(p, posar, edgew)
            + END_BOOST * dists[0][i_s]**2
            + END_BOOST * dists[1][i_g]**2)
            # print("path_cost: %.2f, nx.astar_path_length: %.2f" % (
            #     path_cost(p, posar, edgew),
            #     nx.astar_path_length(g,
            #                       result[0][i_s],
            #                       result[1][i_g],
            #                       heuristic=dist_posar,
            #                       weight='distance'
            #                       )
            # ))
        except nx.exception.NetworkXNoPath:
            c = MAX_COST
        if c < min_c:
            min_c = c
            min_p = p
    # assert min_c != MAX_COST, "no path"
    return min_c, min_p


def plot_path(fig, start, goal, path, posar, edgew):
    xs = [start[0]]
    ys = [start[1]]
    if path:
        for v in path:
            xs.append(posar[v][0])
            ys.append(posar[v][1])
        print("Path cost: %.2f" % path_cost(path, posar, edgew, True))
    xs.append(goal[0])
    ys.append(goal[1])
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.plot(xs, ys, 'k--', linewidth=2, alpha=.6)
    ax.plot([start[0]], [start[1]], 'gx', linewidth=3)
    ax.plot([goal[0]], [goal[1]], 'rx', linewidth=3)
    return ax


def eval(t, evalset, nn, g, ge, pos, posar, edgew, im):
    cost = 0
    unsuccesful = 0
    ne = evalset.shape[0]
    for i in range(ne):
        (c, p) = path(evalset[i, 0], evalset[i, 1], nn, ge, posar, None)
        if c == MAX_COST:
            unsuccesful += 1
        else:
            cost += c
    if t > -1 & t % 10 == 0:
        fig = plt.figure(figsize=[8, 8])
        ax = plot_path(fig, evalset[ne-1, 0], evalset[ne-1, 1], p, posar, edgew)
        plot_graph(fig, ax, g, pos, edgew, im, fname="anim/frame%04d.png" % t)
    return cost / (ne-unsuccesful), unsuccesful


def grad_func(batch, nn, g, ge, posar, edgew):
    out_pos = np.zeros(shape=posar.shape)
    out_edgew = np.zeros(shape=edgew.shape)
    succesful = 0
    batch_cost = 0
    for i_b in range(batch.shape[0]):
        (c, p) = path(batch[i_b, 0], batch[i_b, 1], nn, g, posar, edgew)
        batch_cost += c
        if c != MAX_COST:
            succesful += 1
            coord_p = np.zeros([len(p) + 2, 2])
            coord_p[0, :] = batch[i_b, 0]
            coord_p[1:(1+len(p)), :] = np.array([posar[i_p] for i_p in p])
            coord_p[(1+len(p)), :] = batch[i_b, 1]
            # print(coord_p)
            for i_p in range(len(p)):
                i_cp = i_p + 1
                len_prev = math.sqrt((coord_p[i_cp, 0] - coord_p[i_cp-1, 0]
                             )**2
                            + (coord_p[i_cp, 1] - coord_p[i_cp-1, 1]
                               )**2)
                len_next = math.sqrt((coord_p[i_cp, 0] - coord_p[i_cp+1, 0]
                             )**2
                            + (coord_p[i_cp, 1] - coord_p[i_cp+1, 1]
                               )**2)
                for j in [0, 1]:
                    out_pos[p[i_p], j] += (
                        (coord_p[i_cp, j] - coord_p[i_cp-1, j])
                        / len_prev
                        * (END_BOOST * (coord_p[i_cp, j] - coord_p[i_cp-1, j])
                           / len_prev if i_p == 0
                           else edge_cost_factor(p[i_p-1], p[i_p], edgew))
                        + (coord_p[i_cp, j] - coord_p[i_cp+1, j])
                        / len_next
                        * (END_BOOST * (coord_p[i_cp, j] - coord_p[i_cp+1, j])
                           / len_next if i_p == len(p)-1
                           else edge_cost_factor(p[i_p], p[i_p+1], edgew))
                    )
                if(i_p > 0):
                    if p[i_p-1] < p[i_p]:
                        et = math.exp(-edgew[p[i_p-1], p[i_p]])
                        out_edgew[p[i_p-1], p[i_p]] += (
                            (2. * et / (et + 1) ** 2)
                        ) * len_prev
                    else:
                        et = math.exp(edgew[p[i_p], p[i_p-1]])
                        out_edgew[p[i_p], p[i_p-1]] -= (
                            (2. * et / (et + 1) ** 2)
                        ) * len_prev
                # print(out_pos[p[i_p]])
    return out_pos, out_edgew, batch_cost / batch.shape[0]


def fix(posar_prev, posar, im):
    for i in range(posar.shape[0]):
        if(not is_pixel_free(im, posar[i])):
            posar[i] = posar_prev[i]
