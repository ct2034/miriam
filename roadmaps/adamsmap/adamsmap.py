
from bresenham import bresenham
import imageio
from itertools import product
import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.spatial import Delaunay
from pyflann import *

# Graph
N = 1000

# Paths
nn = 2
MAX_COST = 100000

# Training
ntb = 10  # batch size
nts = 100  # number of batches

# Evaluation
ne = 50  # evaluation set size

im = imageio.imread('simple.png')
im_shape = im.shape


def is_pixel_free(p):
    return min(im[
        int(p[1]),
        int(p[0])
    ]) > 205


def get_random_pos():
    p = np.random.rand(2) * im_shape[0]
    while (not is_pixel_free(p)):
        p = np.random.rand(2) * im_shape[0]
    return p


def dist(sx, sy, gx, gy):
    return math.sqrt((sx-gx)**2+(sy-gy)**2)


def init_graph_posar():
    return np.array([get_random_pos() for _ in range(N)])


def graph_from_posar(posar):
    g = nx.Graph()
    g.add_nodes_from(range(N))
    pos = nx.get_node_attributes(g, 'pos')
    for i in range(N):
        pos[i] = posar[i]
    return g, pos


def make_edges():
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
                if all([is_pixel_free(x) for x in line]):
                    g.add_edge(i, n, distance=dist(
                        sx=posar[i][0], sy=posar[i][1],
                        gx=posar[n][0], gy=posar[n][1]
                    ))


def plot_graph(pos, ax):
    nx.draw_networkx_nodes(g, pos, ax=ax, node_size=20)
    nx.draw_networkx_edges(g, pos, ax=ax, width=0.5, alpha=0.6)
    ax.imshow(im)
    ax.axis('off')
    fig.add_axes(ax)
    plt.show()


def path(start, goal):
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


def plot_path(start, goal, path):
    xs = [start[0]]
    ys = [start[1]]
    for v in path:
        xs.append(pos[v][0])
        ys.append(pos[v][1])
    xs.append(goal[0])
    ys.append(goal[1])
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.plot(xs, ys, 'b')
    return ax


def eval(plot=False):
    cost = 0
    unsuccesful = 0
    for i in range(ne):
        (c, p) = path(evalset[i, 0], evalset[i, 1])
        if c == MAX_COST:
            unsuccesful += 1
        else:
            cost += c
    if plot:
        ax = plot_path(evalset[i, 0], evalset[i, 1], p)
        plot_graph(pos, ax)
    return cost / (ne-unsuccesful)


def grad_func(x, batch):
    out = np.zeros(shape=x.shape)
    for i_b in range(batch.shape[0]):
        (c, p) = path(batch[i_b, 0], batch[i_b, 1])
        if c != MAX_COST:
            coord_p = np.zeros([len(p) + 2, 2])
            coord_p[0, :] = batch[i_b, 0]
            coord_p[1:(1+len(p)), :] = np.array([x[i_p] for i_p in p])
            coord_p[(1+len(p)), :] = batch[i_b, 1]
            for i_p in range(len(p)):
                i_cp = i_p + 1
                for j in [0, 1]:
                    out[p[i_p]] += (
                        (coord_p[i_cp, j] - coord_p[i_cp-1, j])
                        / math.sqrt((coord_p[i_cp, 0] - coord_p[i_cp-1, 0])**2
                                  + (coord_p[i_cp, 1] - coord_p[i_cp-1, 1])**2)
                        + (coord_p[i_cp, j] - coord_p[i_cp+1, j])
                        / math.sqrt((coord_p[i_cp, 0] - coord_p[i_cp+1, 0])**2
                                  + (coord_p[i_cp, 1] - coord_p[i_cp+1, 1])**2)
                    )
    return out


def fix(posar_prev, posar):
    for i in range(posar.shape[0]):
        if(not is_pixel_free(posar[i])):
            posar[i] = posar_prev[i]


evalset = np.array([
    [get_random_pos(), get_random_pos()] for _ in range(ne)])
evalcosts = []

alpha = 0.1
beta_1 = 0.9
beta_2 = 0.9
epsilon = 1

m_t = np.zeros([N, 2])
v_t = np.zeros([N, 2])

for t in range(nts):
    if t == 0:
        posar = init_graph_posar()
    g, pos = graph_from_posar(posar)
    make_edges()
    e_cost = eval()
    print(e_cost)
    evalcosts.append(e_cost)

    batch = np.array([
        [get_random_pos(), get_random_pos()] for _ in range(ntb)])
    # Adam
    # ~~~~
    g_t = grad_func(posar, batch)
    m_t = beta_1*m_t + (1-beta_1)*g_t
    v_t = beta_2*v_t + (1-beta_2)*(g_t*g_t)
    m_cap = m_t / (1-(beta_1**(t+1)))
    v_cap = v_t / (1-(beta_2**(t+1)))
    posar_prev = np.copy(posar)
    posar = posar - np.divide((alpha * m_cap), (np.sqrt(v_cap) + epsilon))


fig = plt.figure()
plt.plot(evalcosts)
fig = plt.figure(figsize=[8, 8])
eval(plot=True)
