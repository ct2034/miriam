
import imageio
from itertools import product
import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.spatial import Delaunay
from pyflann import *

#Graph
N = 1000

#Paths
nn = 3

#Training
ntb = 100 #number of batches
nts = 100 #batch size

#Evaluation
ne = 50 #evaluation set size

im = imageio.imread('fake.png')
im_shape = im.shape

def is_pixel_free(p):
    return min(im[
        int(p[1]),
        int(p[0])
        ]) < 255

def get_random_pos():
    p = np.random.rand(2) * im_shape[0]
    while (is_pixel_free(p)):
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

posar = init_graph_posar()
g, pos = graph_from_posar(posar)

def make_edges():
    tri = Delaunay(posar)
    (indptr, indices) = tri.vertex_neighbor_vertices
    for i in range(N):
        neigbours = indices[indptr[i]:indptr[i+1]]
        for n in neigbours:
            if i < n:
                g.add_edge(i, n, distance=dist(
                    sx=posar[i][0], sy=posar[i][1], gx=posar[n][0], gy=posar[n][1]
                    ))
make_edges()

def plot_graph(pos):
    nx.draw_networkx_nodes(g, pos, node_size=30)
    nx.draw_networkx_edges(g, pos, width=0.5, alpha=0.6)
    # plt.axis('off')
    # ax = plt.axes()
    plt.imshow(im)
    plt.axis('off')
    plt.show()
# plot_graph(pos)

def path(start, goal):
    flann = FLANN()
    result, dists = flann.nn(
        posar, np.array([start, goal]), nn,
        algorithm="kmeans", branching=32, iterations=7, checks=16)
    min_c = 1000
    min_p = None
    for (i_s, i_g) in product(range(nn), range(nn)):
        c = nx.shortest_path_length(g, result[0][i_s], result[1][i_g], weight='distance')
        if c < min_c:
            min_p = nx.shortest_path(g, result[0][i_s], result[1][i_g], weight='distance')
            min_c = c + dists[0][i_s] + dists[1][i_g]
    assert min_c != 1000, "no path"
    return min_c, min_p

evalset = np.array( [
    [get_random_pos(), get_random_pos()] for _ in range(ne) ])

def plot_path(start, goal, path):
    xs = [start[0]]
    ys = [start[1]]
    for v in path:
        xs.append(pos[v][0])
        ys.append(pos[v][1])
    xs.append(goal[0])
    ys.append(goal[1])
    plt.plot(xs, ys, 'b')

def eval():
    cost = 0
    for i in range(ne):
        (c, p) = path(evalset[i, 0], evalset[i, 1])
        cost += c
    plt.figure(figsize=[8, 8])
    plot_path(evalset[i, 0], evalset[i, 1], p)
    plot_graph(pos)
    return cost

# for i_t in range(nts):

print(eval())
