#!/usr/bin/env python3
import imageio
from itertools import combinations
from math import sqrt
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pickle
from random import random
import sys

from adamsmap import (
    dist,
    get_random_pos,
    grad_func,
    graphs_from_posar,
    init_graph_posar_edgew,
    make_edges,
    MAX_COST,
    path
)


def eval(batch, nn, g, posar, edgew):
    sim_paths = []
    for i_b in range(batch.shape[0]):
        (c, p) = path(batch[i_b, 0], batch[i_b, 1], nn, g, posar, edgew)
        if c < MAX_COST:
            coord_p = np.zeros([len(p) + 2, 2])
            coord_p[0, :] = batch[i_b, 0]
            coord_p[1:(1+len(p)), :] = np.array([posar[i_p] for i_p in p])
            coord_p[(1+len(p)), :] = batch[i_b, 1]
            current = coord_p[0]
            sim_path = []
            i = 0
            while dist(current, batch[i_b, 1]) > v:
                sim_path.append(current)
                next_p = coord_p[i + 1]
                d_next_p = dist(current, next_p)
                if d_next_p > v:
                    delta = v * (next_p - current) / d_next_p
                    current += delta
                else:  # d_next_p < v
                    rest = v - d_next_p
                    assert(rest < v)
                    nnext_p = coord_p[i + 2]
                    d_nnext_p = dist(nnext_p, next_p)
                    delta = rest * (nnext_p - next_p) / d_nnext_p
                    current = next_p + delta
                    i += 1
            sim_paths.append(np.array(sim_path))
        else:
            print("Path failed !!")
            sim_paths.append(np.array([batch[i_b, 0]]))
    # for p in sim_paths:
    #     plt.plot(p[:, 0], p[:, 1])
    # plt.show()
    sim_paths_coll = None

    ended = [False for _ in range(agents)]
    waiting = [False for _ in range(agents)]
    i_per_agent = [0 for _ in range(agents)]
    t_end = [0 for _ in range(agents)]
    timeslice = np.zeros([agents, 2])
    while not all(ended):
        ended = [sim_paths[i].shape[0] - 1 == i_per_agent[i]
                 for i in range(agents)]
        for i_a in range(agents):
            timeslice[i_a, :] = sim_paths[i_a][i_per_agent[i_a]]
            if ended[i_a]:
                t_end[i_a] = i_per_agent[i_a]
        if sim_paths_coll is None:
            sim_paths_coll = np.array([timeslice, ])
        else:
            sim_paths_coll = np.append(sim_paths_coll,
                                       np.array([timeslice, ]),
                                       axis=0)
        for (a, b) in combinations(range(agents), r=2):
            waiting = [False for _ in range(agents)]
            if dist(sim_paths[a][i_per_agent[a]],
                    sim_paths[b][i_per_agent[b]]) < agent_d:
                waiting[min(a, b)] = True
        i_per_agent = [i_per_agent[i] + (1 if not waiting[i]
                                         and not ended[i]
                                         else 0)
                       for i in range(agents)]

    # from mpl_toolkits.mplot3d import Axes3D
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # for i in range(agents):
    #     ax.plot(sim_paths_coll[:,i,0],
    #             sim_paths_coll[:,i,1],
    #             range(sim_paths_coll.shape[0]))
    # plt.show()
    return sum(t_end)


if __name__ == '__main__':
    fname = sys.argv[1]
    with open(fname, "rb") as f:
        store = pickle.load(f)

    agents = 20
    agent_d = 60  # disk diameter
    v = 1
    nn = 1
    posar = store['posar']
    edgew = store['edgew']
    N = posar.shape[0]
    im = imageio.imread(fname.split("_")[0]+".png")
    g, ge, pos = graphs_from_posar(N, posar)
    make_edges(N, g, ge, posar, edgew, im)
    batch = np.array([
        [get_random_pos(im), get_random_pos(im)] for _ in range(agents)])

    print(eval(batch, nn, ge, posar, edgew))
    edgew_compare = np.ones([N, N])
    g_compare = nx.Graph()
    g_compare.add_nodes_from(range(N))
    for e in nx.edges(ge):
        g_compare.add_edge(e[0],
                           e[1],
                           distance=dist(posar[e[0]], posar[e[1]]))
    print(eval(batch, nn, g_compare, posar, edgew_compare))
