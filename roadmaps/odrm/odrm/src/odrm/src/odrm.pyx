#!/usr/bin/env python3
from bresenham import bresenham
from functools import reduce
from itertools import product
import math
from matplotlib import cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.spatial import Delaunay
from pyflann import FLANN

MAX_COST = 100000
END_BOOST = 3.


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
    try:
        return math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)
    except Exception as e:
        print(a)
        print(b)
        print(e)

def edge_cost_factor(a, b, edgew):
    if a < b:
        c = edgew[a, b]
    else:  # b < a
        c = - edgew[b, a]
    return 2 / (1+math.exp(-c)) + 1


def path_cost(p, posar, edgew, prin=False):
    if edgew is not None:  # ge
        return reduce(lambda x, y: x+y,
                      [dist(posar[p[i]], posar[p[i+1]])
                       + edge_cost_factor(p[i], p[i+1], edgew)
                       for i in range(len(p)-1)], 0.)
    else:
        return reduce(lambda x, y: x+y,
                      [dist(posar[p[i]], posar[p[i+1]])
                       for i in range(len(p)-1)], 0.)
    # TODO: use graph weight


def init_random_edgew(N):
    return np.triu(np.random.normal(loc=0, scale=0.1, size=(N, N)), 1)


def init_graph_posar_edgew(im, N):
    global posar
    posar = np.array([get_random_pos(im) for _ in range(N)])
    edgew = init_random_edgew(N)
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


def get_edge_statistics(g, posar):
    edge_lengths = []
    for e in g.edges:
        edge_lengths.append(dist(posar[e[0]],
                                 posar[e[1]]))
    return np.mean(edge_lengths), np.std(edge_lengths)


def plot_graph(fig, ax, g, pos, edgew, im, fname='', edgecol=True, show=True):
    nx.draw_networkx_nodes(g, pos, ax=ax, node_size=15, node_color='k')

    def show_edge(e):
        if e[0] < e[1]:
            return edgew[e[0], e[1]] > 0
        else:  # e[1] < e[0]
            return edgew[e[1], e[0]] < 0

    edges = list(filter(show_edge, g.edges()))
    if edgecol:
        colperval = lambda val: cm.brg(.5 * abs(val) + .5)
    else:
        colperval = lambda val: cm.binary(abs(val))
    edge_colors = [colperval(val) for val in
                   map(lambda x: edgew[x[0], x[1]] if x[0] < x[1]
                       else edgew[x[1], x[0]], edges)]

    nx.draw_networkx_edges(g, pos, ax=ax, edgelist=edges,
                           width=0.7, edge_color=edge_colors)
    ax.imshow(im)
    ax.axis('off')
    fig.add_axes(ax)
    if(show):
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
        if nn == 1:
            start_v = result[0]
            goal_v = result[1]
        else:
            start_v = result[0][i_s]
            goal_v = result[1][i_g]
        p = vertex_path(g, start_v, goal_v, posar)
        if p is None:
            c = MAX_COST
        else:
            ds = dist(start, posar[p[0]])
            dg = dist(goal, posar[p[-1]])
            c = (path_cost(p, posar, edgew)
                 + END_BOOST * (ds**2 + ds)
                 + END_BOOST * (dg**2 + dg))
        if c < min_c:
            min_c = c
            min_p = p
    return min_c, min_p


def vertex_path(g, start_v, goal_v, posar):
    def dist_posar(an, bn):
        return dist(posar[an], posar[bn])
    try:
        return nx.astar_path(g,
                             start_v,
                             goal_v,
                             heuristic=dist_posar,
                             weight='distance')
    except nx.exception.NetworkXNoPath:
        return None


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


def eval(t, evalset, nn, g, ge, pos, posar, edgew, im, plot=True):
    cost = 0
    unsuccesful = 0
    ne = evalset.shape[0]
    for i in range(ne):
        (c, p) = path(evalset[i, 0], evalset[i, 1], nn, ge, posar, None)
        if c == MAX_COST:
            unsuccesful += 1
        else:
            cost += c
    if plot and t > -1 and t % 10 == 0:
        print("plotting ..")
        fig = plt.figure(figsize=[8, 8])
        ax = plot_path(fig, evalset[ne-1, 0], evalset[ne-1, 1], p, posar, edgew)
        plot_graph(fig, ax, g, pos, edgew, im, fname="anim/frame%04d.png" % t)
    return cost / (ne-unsuccesful), unsuccesful


def grad_func(batch, nn, g, ge, posar_, edgew):
    global posar
    posar = posar_
    out_pos = np.zeros(shape=posar.shape)
    out_edgew = np.zeros(shape=edgew.shape)
    succesful = 0
    batch_cost = 0
    for i_b in range(batch.shape[0]):
        (c, p) = path(batch[i_b, 0], batch[i_b, 1], nn, g, posar, edgew)
        if c != MAX_COST:
            succesful += 1
            batch_cost += c
            coord_p = np.zeros([len(p) + 2, 2])
            coord_p[0, :] = batch[i_b, 0]
            coord_p[1:(1+len(p)), :] = np.array([posar[i_p] for i_p in p])
            coord_p[(1+len(p)), :] = batch[i_b, 1]
            # print(coord_p)
            for i_p in range(len(p)):
                i_cp = i_p + 1
                len_prev = dist(coord_p[i_cp], coord_p[i_cp-1])
                len_next = dist(coord_p[i_cp], coord_p[i_cp+1])
                for j in [0, 1]:
                    out_pos[p[i_p], j] += (
                          (0 if i_p == 0 else 1)
                        * (coord_p[i_cp, j] - coord_p[i_cp-1, j])
                        / len_prev
                        + (0 if i_p == len(p)-1 else 1)
                        * (coord_p[i_cp, j] - coord_p[i_cp+1, j])
                        / len_next
                    )
                    if i_p == 0:  # tail costs start
                        out_pos[p[i_p], j] += (
                            END_BOOST
                            * (coord_p[i_cp, j] - coord_p[i_cp-1, j])
                            * (1. / len_prev + 2)
                        )
                    if i_p == len(p)-1:  # tail costs goal
                        out_pos[p[i_p], j] += (
                            END_BOOST
                            * (coord_p[i_cp, j] - coord_p[i_cp+1, j])
                            * (1. / len_next + 2)
                        )
                if(i_p > 0):
                    if p[i_p-1] < p[i_p]:
                        et = math.exp(-edgew[p[i_p-1], p[i_p]])
                        out_edgew[p[i_p-1], p[i_p]] += (
                            (2. * et / (et + 1) ** 2)
                        )
                    else:
                        et = math.exp(edgew[p[i_p], p[i_p-1]])
                        out_edgew[p[i_p], p[i_p-1]] -= (
                            (2. * et / (et + 1) ** 2)
                        )
    return out_pos, out_edgew, batch_cost


def fix(posar_prev, posar, im):
    for i in range(posar.shape[0]):
        if(not is_pixel_free(im, posar[i])):
            posar[i] = posar_prev[i]
