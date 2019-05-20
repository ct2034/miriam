#!/usr/bin/env python3
import imageio
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pickle
import random
import sys

from adamsmap import (
    get_random_pos,
    graphs_from_posar,
    init_graph_posar_edgew,
    init_random_edgew,
    make_edges,
    plot_graph,
    eval,
    grad_func,
    fix
    )
from adamsmap_filename_verification import (
    is_eval_file,
    resolve_mapname
    )
from eval_disc import (
    simulate_paths_indep
    )


def random_pos_on_edge(g, posar):
    NE = nx.number_of_edges(g)
    d = random.random() * NE
    edges = nx.to_edgelist(g)
    edge = None
    for i, e in enumerate(edges):
        if i == int(d):
            edge = e
    assert edge is not None, "failed to fine edge"
    pos = np.array(posar[edge[0]]) + (d - int(d)) * (
        np.array(posar[edge[1]]) - np.array(posar[edge[0]]))
    return pos


if __name__ == '__main__':
    """movement speed"""
    v = .2

    fname = sys.argv[1]
    with open(fname, "rb") as f:
        assert is_eval_file(fname), "Please call with eval file"
        store = pickle.load(f)
    im = imageio.imread(resolve_mapname(fname))
    random.seed(0)

    posar = store['posar']
    edgew = store['edgew']
    N = posar.shape[0]
    g, ge, pos = graphs_from_posar(N, posar)
    make_edges(N, g, ge, posar, edgew, im)

    edgew_new = init_random_edgew(N)
    _, ge_new, _ = graphs_from_posar(N, posar)
    make_edges(N, g, ge_new, posar, edgew_new, im)

    for agents in [100, 300, 1000]:
        batch = np.array([
            [random_pos_on_edge(g, posar), random_pos_on_edge(g, posar)] for _ in range(agents)])

        # plot
        fig = plt.figure(figsize=[10, 10])
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.plot(batch[:, 0, 0], batch[:, 0, 1], 'og')
        ax.plot(batch[:, 1, 0], batch[:, 1, 1], 'xr')

        paths_undir = simulate_paths_indep(batch, edgew, g, 1, posar, v)
        t_undir = sum(map(lambda p: len(p) * v, paths_undir))
        print("t_undir: {0}".format(t_undir))

        paths_dir = simulate_paths_indep(batch, edgew, ge, 1, posar, v)
        t_dir = sum(map(lambda p: len(p) * v, paths_dir))
        print("t_dir: {0}".format(t_dir))

        paths_rand = simulate_paths_indep(batch, edgew_new, ge_new, 1, posar, v)
        t_rand = sum(map(lambda p: len(p) * v, paths_rand))
        print("t_rand: {0}".format(t_rand))

        for p in paths_rand:
            ax.plot(p[:, 0], p[:, 1], alpha=.6)

        plot_graph(fig, ax, g, pos, edgew_new, im, '', False)

