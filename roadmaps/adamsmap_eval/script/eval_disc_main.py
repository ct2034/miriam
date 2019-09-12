#!/usr/bin/env python2
import csv
import logging
import pickle
import random
import sys
from functools import reduce
from itertools import combinations, product

import imageio
import networkx as nx
import numpy as np
from adamsmap.adamsmap import (
    dist,
    get_edge_statistics,
    get_random_pos,
    graphs_from_posar,
    is_pixel_free,
    make_edges,
    vertex_path
)
from bresenham import bresenham
from scipy.spatial import Delaunay

from adamsmap_eval.filename_verification import (
    is_result_file,
    is_eval_file,
    resolve_mapname,
    resolve
)
from adamsmap_eval.eval_disc import (
    get_unique_batch,
    eval_disc,
    write_csv
)

logging.basicConfig(level=logging.INFO)

# how bigger than its size should the robot sense?
SENSE_FACTOR = 0.

if __name__ == '__main__':
    random.seed(0)
    fname = sys.argv[1]
    with open(fname, "rb") as f:
        assert is_result_file(fname), "Please call with result file"
        store = pickle.load(f)

    agent_ns = range(10, 200, 30)
    # agent_ns = [10, 20, 40]
    res = {}
    for ans in agent_ns:
        res[ans] = {}
        res[ans]["undir"] = []
        res[ans]["rand"] = []
        res[ans]["paths_ev"] = []
        res[ans]["paths_undirected"] = []
        res[ans]["paths_random"] = []

    posar = store['posar']
    N = posar.shape[0]
    edgew = store['edgew']
    im = imageio.imread(resolve_mapname(fname))
    __, ge, pos = graphs_from_posar(N, posar)
    make_edges(N, __, ge, posar, edgew, im)
    logging.info("edge stats: " + str(get_edge_statistics(ge, posar)))

    for agents, agent_diameter, i_trial in product(
            agent_ns, [10], range(1)):
        logging.info("agents: " + str(agents))
        logging.info("agent_diameter: " + str(agent_diameter))
        v = .2
        nn = 1
        batch = get_unique_batch(N, agents)
        cost_ev, paths_ev = eval_disc(batch, ge,
                                      posar, agent_diameter, v)
        write_csv(agents, paths_ev, "ev-our", i_trial, fname)

        edgew_undirected = np.ones([N, N])
        g_undirected = nx.Graph()
        g_undirected.add_nodes_from(range(N))
        for e in nx.edges(ge):
            g_undirected.add_edge(e[0],
                                  e[1],
                                  distance=dist(posar[e[0]], posar[e[1]]))
        cost_undirected, paths_undirected = (eval_disc(batch, g_undirected,
                                                       posar, agent_diameter, v))
        write_csv(agents, paths_undirected, "undirected", i_trial, fname)

        g_random = nx.Graph()
        g_random.add_nodes_from(range(N))
        posar_random = np.array([get_random_pos(im) for _ in range(N)])
        b = im.shape[0]
        fakenodes1 = np.array(np.array(list(
            product([0, b], np.linspace(0, b, 6)))))
        fakenodes2 = np.array(np.array(list(
            product(np.linspace(0, b, 6), [0, b]))))
        tri = Delaunay(np.append(posar_random, np.append(
            fakenodes1, fakenodes2, axis=0), axis=0
                                 ))
        (indptr, indices) = tri.vertex_neighbor_vertices
        for i_n in range(N):
            neigbours = indices[indptr[i_n]:indptr[i_n + 1]]
            for n in neigbours:
                if (i_n < n) & (n < N):
                    line = bresenham(
                        int(posar_random[i_n][0]),
                        int(posar_random[i_n][1]),
                        int(posar_random[n][0]),
                        int(posar_random[n][1])
                    )
                    if all([is_pixel_free(im, x) for x in line]):
                        g_random.add_edge(i_n, n,
                                          distance=dist(posar_random[i_n],
                                                        posar_random[n]))
                        g_random.add_edge(n, i_n,
                                          distance=dist(posar_random[i_n],
                                                        posar_random[n]))
        cost_random, paths_random = eval_disc(batch, g_random,
                                              posar_random, agent_diameter, v)
        write_csv(agents, paths_random, "random", i_trial, fname)

        logging.info("our: %d, undir: %d, (our-undir)/our: %.3f%%" %
                      (cost_ev, cost_undirected,
                       100. * float(cost_ev - cost_undirected) / cost_ev))
        logging.info("our: %d, rand: %d, (our-rand)/our: %.3f%%\n-----" %
                      (cost_ev, cost_random,
                       100. * float(cost_ev - cost_random) / cost_ev))

        res[agents]["undir"].append(100. * float(
            cost_ev - cost_undirected) / cost_ev)
        res[agents]["rand"].append(100. * float(
            cost_ev - cost_random) / cost_ev)
        res[agents]["paths_ev"].append(paths_ev)
        res[agents]["paths_undirected"].append(paths_undirected)
        res[agents]["paths_random"].append(paths_random)

    fname_write = sys.argv[1] + ".eval"
    assert is_eval_file(fname_write), "Please write " \
                                      "results to eval file (ending with pkl.eval)"
    with open(fname_write, "wb") as f:
        pickle.dump(res, f)
