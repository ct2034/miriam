#!/usr/bin/env python3
from bresenham import bresenham
import imageio
from itertools import combinations, product
import networkx as nx
import numpy as np
from scipy.spatial import Delaunay
import pickle
import sys

from adamsmap import (
    dist,
    get_random_pos,
    grad_func,
    graphs_from_posar,
    init_graph_posar_edgew,
    is_pixel_free,
    make_edges,
    MAX_COST,
    path
)
from adamsmap_filename_verification import (
    is_result_file,
    is_eval_file
)


def eval_disc(batch, nn, g, posar, edgew):
    """
    Evaluating a given graph by simulating disc-shaped robots to travel through it. The robots move to their individual
        goals and whenever they would collide, they stop based on an arbitrary priority.

    :param batch: a batch of start / goal pairs for agents
    :param nn: how many nearest neighbours to consider when path-planning
    :param g: the graph to plan on (undirected)
    :param posar: poses of the graph nodes
    :param edgew: edge weights of the agents (determining the direction of the edges)
    :return: sum of costs, paths
    """
    sim_paths = []
    for i_b in range(batch.shape[0]):
        (c, p) = path(batch[i_b, 0], batch[i_b, 1], nn, g, posar, edgew)
        if c < MAX_COST:
            coord_p = np.zeros([len(p) + 2, 2])
            coord_p[0, :] = batch[i_b, 0]
            coord_p[1:(1+len(p)), :] = np.array([posar[i_p] for i_p in p])
            coord_p[(1+len(p)), :] = batch[i_b, 1]
            goal = batch[i_b, 1]
            sim_path = move_to_next_point(goal, coord_p)
            sim_paths.append(np.array(sim_path))
        else:
            print("Path failed !!")
            sim_paths.append(np.array([batch[i_b, 0]]))
    sim_paths_coll = None

    ended = [False for _ in range(agents)]
    waiting = [False for _ in range(agents)]
    i_per_agent = [0 for _ in range(agents)]
    t_end = [0 for _ in range(agents)]
    time_slice = np.zeros([agents, 2])
    while not all(ended):
        ended = [sim_paths[i].shape[0] - 1 == i_per_agent[i]
                 for i in range(agents)]
        for i_a in range(agents):
            time_slice[i_a, :] = sim_paths[i_a][i_per_agent[i_a]]
            if ended[i_a]:
                t_end[i_a] = i_per_agent[i_a]
        if sim_paths_coll is None:
            sim_paths_coll = np.array([time_slice, ])
        else:
            sim_paths_coll = np.append(sim_paths_coll,
                                       np.array([time_slice, ]),
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
    return sum(t_end), sim_paths_coll


def move_to_next_point(goal, coord_p):
    """
    Simulate one agent path through coordinates.

    :param goal: goal coordinates for this agent
    :param coord_p: the coordinates for the path to be followed
    :return: the path in coordinates
    """
    sim_path = []
    i = 0
    current = coord_p[0]
    while dist(current, goal) > v:
        sim_path.append(current)
        next_p = coord_p[i + 1]
        d_next_p = dist(current, next_p)
        if d_next_p > v:
            delta = v * (next_p - current) / d_next_p
            current += delta
        else:  # d_next_p < v
            rest = v - d_next_p
            assert (rest < v)
            next_node_p = coord_p[i + 2]
            d_next_node_p = dist(next_node_p, next_p)
            delta = rest * (next_node_p - next_p) / d_next_node_p
            current = next_p + delta
            i += 1
    return sim_path


if __name__ == '__main__':
    fname = sys.argv[1]
    with open(fname, "rb") as f:
        assert is_result_file(fname), "Please call with result file"
        store = pickle.load(f)

    agent_ns = [10, 20, 40]
    res = {}
    for ans in agent_ns:
        res[ans] = {}
        res[ans]["undir"] = []
        res[ans]["rand"] = []
        res[ans]["paths_ev"] = []
        res[ans]["paths_undirected"] = []
        res[ans]["paths_random"] = []

    for agents, agent_d, i in product(
        [20], [20], range(2)):
        print("agents: " + str(agents))
        print("agent_d: " + str(agent_d))
        v = 1
        nn = 1
        posar = store['posar']
        edgew = store['edgew']
        N = posar.shape[0]
        im = imageio.imread("maps/"+fname.split("_")[0].split("/")[-1]+".png")
        g, ge, pos = graphs_from_posar(N, posar)
        make_edges(N, g, ge, posar, edgew, im)
        batch = np.array([
            [get_random_pos(im), get_random_pos(im)] for _ in range(agents)])
        cost_ev, paths_ev = eval_disc(batch, nn, ge, posar, edgew)

        edgew_undirected = np.ones([N, N])
        g_undirected = nx.Graph()
        g_undirected.add_nodes_from(range(N))
        for e in nx.edges(ge):
            g_undirected.add_edge(e[0],
                               e[1],
                               distance=dist(posar[e[0]], posar[e[1]]))
        cost_undirected, paths_undirected = (eval_disc(batch, nn, g_undirected, posar, edgew_undirected))

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
        for i in range(N):
            neigbours = indices[indptr[i]:indptr[i+1]]
            for n in neigbours:
                if i < n & n < N:
                    line = bresenham(
                        int(posar_random[i][0]),
                        int(posar_random[i][1]),
                        int(posar_random[n][0]),
                        int(posar_random[n][1])
                    )
                    # print(list(line))
                    if all([is_pixel_free(im, x) for x in line]):
                        g_random.add_edge(i, n,
                                   distance=dist(posar_random[i], posar_random[n]))
                        g_random.add_edge(n, i,
                                   distance=dist(posar_random[i], posar_random[n]))
        cost_random, paths_random = eval_disc(batch, nn, g_random, posar_random, edgew_undirected)

        print("our: %d, undir: %d, (our-undir)/our: %.3f%%" %
              (cost_ev, cost_undirected,
               100.*float(cost_ev-cost_undirected)/cost_ev))
        print("our: %d, rand: %d, (our-rand)/our: %.3f%%\n-----" %
              (cost_ev, cost_random,
               100.*float(cost_ev-cost_random)/cost_ev))

        res[agents]["undir"].append(100.*float(cost_ev-cost_undirected)/cost_ev)
        res[agents]["rand"].append(100.*float(cost_ev-cost_random)/cost_ev)
        res[agents]["paths_ev"].append(paths_ev)
        res[agents]["paths_undirected"].append(paths_undirected)
        res[agents]["paths_random"].append(paths_random)

    fname_write = sys.argv[1] + ".eval"
    assert is_eval_file(fname_write), "Please write results to eval file (ending with pkl.eval)"
    with open(fname_write, "wb") as f:
        pickle.dump(res, f)
