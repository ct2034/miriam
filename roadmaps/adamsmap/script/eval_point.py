#!/usr/bin/env python2
import csv

import imageio
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pickle
import random
import sys

from adamsmap.adamsmap import (
    graphs_from_posar,
    make_edges,
    plot_graph
    )
from adamsmap_filename_verification import (
    is_result_file,
    resolve_mapname
    )
from eval_disc import (
    simulate_one_path
    )


def random_pos_and_edge_on_edge(ge, posar):
    """
    Return a random position on and edge and that edge
    :param ge: graph
    :param posar: position array
    :return: the random potition as np.array and the edge it is on
    """
    NE = nx.number_of_edges(ge)
    d = random.random() * NE
    edges = nx.to_edgelist(ge)
    edge = None
    for i, e in enumerate(edges):
        if i == int(d):
            edge = e
            break
    assert edge is not None, "failed to fine edge"
    pos = np.array(posar[edge[0]]) + (d - int(d)) * (
        np.array(posar[edge[1]]) - np.array(posar[edge[0]]))
    return pos, edge


def get_random_route(g, posar):
    """
    Get a start/goal pair that each lie on an edge
    :param g: graph
    :param posar: position array
    :return: start / goal coordinates + next and prev node numbers
    """
    start_coords, start_edge = random_pos_and_edge_on_edge(g, posar)
    goal_coords, goal_edge = random_pos_and_edge_on_edge(g, posar)
    return start_coords, goal_coords, (start_edge[1], goal_edge[0])


def path(g, batchpart):
    """
    plan a path
    :param g: the graph
    :param batchpart: the part of the batch, containing start / goal coordinats
        + next and prev node numbers
    :return: a path in node numbers
    """
    p = nx.shortest_path(
        g,
        int(batchpart[2, 0]),
        int(batchpart[2, 1]))
    return p


def coord_path(g, batchpart, posar, v):
    """
    plan a path in coordinates
    :param g: the graph
    :param batchpart: the part of the batch, containing start / goal coordinats
        + next and prev node numbers
    :param posar: position array
    :param v: velocity to travel in
    :return: a path in coordinates
    """
    try:
        p = path(g, batchpart)
    except nx.exception.NetworkXNoPath:
        print(batchpart)
        return []
    if p is None:
        return []
    coord_p = np.zeros([len(p) + 2, 2])
    coord_p[0, :] = batchpart[0]
    coord_p[1:(1 + len(p)), :] = np.array([posar[i_p] for i_p in p])
    coord_p[(1 + len(p)), :] = batchpart[1]
    goal = batchpart[1].copy()
    sim_path = np.array(simulate_one_path(goal, coord_p, v))
    assert (sim_path[0, :] == batchpart[0]).all()
    assert (sim_path[-1, :] == batchpart[1]).all()
    return sim_path


def get_dead_ends(g):
    N = nx.number_of_nodes(g)
    inedges = np.zeros(N)
    outedges = np.zeros(N)
    for n in nx.nodes(g):
        outedges[n] = 1
        for ne in nx.neighbors(g, n):
            inedges[ne] = 1
    noinedges = list(filter(lambda i: inedges[i] == 0, range(N)))
    nooutedges = list(filter(lambda i: outedges[i] == 0, range(N)))
    assert (len(set(noinedges).intersection(nooutedges))
            == 0), "a node with no edge ?!"
    return noinedges, nooutedges


def fix_dead_ends(g):
    for i in range(100):
        dei, deo = get_dead_ends(g)
        nde = len(dei + deo)
        if nde == 0:
            return
        fix = random.choice(dei + deo)
        g_old = g.copy()
        if fix in dei:  # we have no in edges
            fix2 = random.choice(list(nx.neighbors(g, fix)))
            g.remove_edge(fix, fix2)
            g.add_edge(fix2, fix)
        else:  # no out edges
            assert fix in deo, "must have no out edge, then"
            candidates_fix2 = []
            for n in nx.nodes(g):
                for ne in nx.neighbors(g, n):
                    if ne == fix:
                        candidates_fix2.append(n)
            fix2 = random.choice(candidates_fix2)
            g.remove_edge(fix2, fix)
            g.add_edge(fix, fix2)

        dei2, deo2 = get_dead_ends(g)
        nde2 = len(dei2 + deo2)
        print(nde2)
        if nde2 >= nde:  # if it didn't help
            g = g_old  # reset to old graph


def directional_consensus(g_undir, batch):
    """
    rebuild a directed graph based on the directional preference from a set of
    paths
    :param g_undir: base graph, undirected
    :param batch: path start and goal poses
    :return: the newly directional graph
    """
    first_paths = []
    agents = batch.shape[0]
    for i_a in range(agents):
        p = path(g_undir, batch[i_a])
        first_paths.append(p)
    direction_consensus = build_consensus(first_paths, nx.edges(g_undir))
    g_dir_again = nx.DiGraph()
    g_dir_again.add_nodes_from(nx.nodes(g_undir))
    for e, dir in direction_consensus.items():
        if dir == 0:  # solving undecided edges by chance ..
            dir = random.choice([1, -1])
        if dir > 0:
            g_dir_again.add_edge(e[0], e[1])
        elif dir < 0:
            g_dir_again.add_edge(e[1], e[0])
    fix_dead_ends(g_dir_again)
    return g_dir_again


def build_consensus(first_paths, edges):
    """
    based on the given paths, build a directional consensos for the given edges
    :param first_paths: a set of paths to decide on the directions
    :param edges: edges for consensus to be built from
    :return: a dict containing numbers > 0 for this direction, < 0 for inverse
    """
    direction_consensus = {}
    for e in edges:
        direction_consensus[tuple(sorted((e[0], e[1])))] = 0
    for p in first_paths:
        prev = None
        for n in p:
            if prev is not None:
                if sorted((prev, n)) == [prev, n]:  # this direction
                    direction_consensus[tuple(sorted((prev, n)))] += 1
                else:  # other direction
                    direction_consensus[tuple(sorted((prev, n)))] += -1
            prev = n
    return direction_consensus


if __name__ == '__main__':
    """movement speed"""
    _v = .2
    """n of nearest neighbours"""
    _nn = 2

    _fname = sys.argv[1]
    with open(_fname, "rb") as f:
        assert is_result_file(_fname), "Please call with result file"
        store = pickle.load(f)
    im = imageio.imread(resolve_mapname(_fname))

    _posar = store['posar']
    _edgew = store['edgew']
    _N = _posar.shape[0]
    _, _g_dir, _pos = graphs_from_posar(_N, _posar)
    _g_undir = nx.to_undirected(_g_dir)
    assert isinstance(_g_dir, nx.DiGraph)
    assert isinstance(_g_undir, nx.Graph)
    make_edges(_N, _, _g_dir, _posar, _edgew, im)

    # write graph to csv ############################################
    nx.write_adjlist(_g_dir, "graph_adjlist.csv")                   #
    with open("graph_pos.csv", 'w') as f_csv:                       #
        writer = csv.writer(f_csv, delimiter=' ')                   #
        for i_a in range(_N):                                       #
            writer.writerow(_posar[i_a])                            #
    #################################################################

    for agents in [10, 30, 100, 300, 1000]:
        print("%d agents" % agents)

        random.seed(0)
        _batch = []
        for i_a in range(agents):
            _batch.append(get_random_route(_g_dir, _posar))
        _batch = np.array(_batch)

        paths_dir = []
        unsucc_dir = 0
        for i_a in range(agents):
            p = coord_path(_g_dir, _batch[i_a], _posar, _v)
            if len(p) == 0:
                unsucc_dir += 1
            else:
                paths_dir.append(p)
        t_dir = sum(map(lambda p: len(p) * _v, paths_dir))
        print("t_dir: {0:.2f}".format(t_dir/agents))
        print("unsucc_dir: {0}".format(unsucc_dir))
        print("-"*20)

        g_dir_again = directional_consensus(_g_undir, _batch)
        paths_undir = []
        unsucc_undir = 0
        for i_a in range(agents):
            p = coord_path(g_dir_again, _batch[i_a], _posar, _v)
            if len(p) == 0:
                unsucc_undir += 1
                unsucc_i_a = i_a
            else:
                paths_undir.append(p)
        t_undir = sum(map(lambda p: len(p) * _v, paths_undir))
        print("t_undir: {0:.2f}".format(t_undir/agents))
        print("unsucc_undir: {0}".format(unsucc_undir))
        print("-"*40)

        # plot
        fig1 = plt.figure(figsize=[10, 10])
        ax1 = plt.Axes(fig1, [0., 0., 1., 1.])
        fig1.suptitle('directed', fontsize=20)
        ax1.plot(_batch[:, 0, 0], _batch[:, 0, 1], 'og')  # start: o green
        ax1.plot(_batch[:, 1, 0], _batch[:, 1, 1], 'xr')  # goal: x red
        #      o--> ---> \
        #                \-> x

        for p in paths_dir:
            ax1.plot(p[:, 0], p[:, 1], alpha=.8)

        plot_graph(fig1, ax1, _g_dir, _pos, _edgew, im, '', False, False)

        # plot
        fig2 = plt.figure(figsize=[10, 10])
        ax2 = plt.Axes(fig2, [0., 0., 1., 1.])
        fig2.suptitle('undirected', fontsize=20)
        ax2.plot(_batch[unsucc_i_a, 0, 0], _batch[unsucc_i_a, 0, 1], 'og')
        ax2.plot(_batch[unsucc_i_a, 1, 0], _batch[unsucc_i_a, 1, 1], 'xr')

        for p in paths_undir:
            ax2.plot(p[:, 0], p[:, 1], alpha=.8)

        plot_graph(fig2, ax2, g_dir_again, _pos, _edgew, im, '', False, False)

        plt.show()
