#!/usr/bin/env python2
import itertools
import logging
import numpy as np
import os
import pickle
import random
import signal
import sys
import time
from enum import Enum, unique
from math import sqrt

import coloredlogs
import imageio
import networkx as nx
from adamsmap.adamsmap import is_pixel_free
from adamsmap_eval.eval_disc import (
    get_unique_batch,
    eval_disc,
    graphs_from_posar,
    make_edges
)
from adamsmap_eval.filename_verification import (
    get_graph_csvs,
    get_graph_undir_csv,
    is_result_file,
    resolve_mapname,
    get_basename_wo_extension
)
from bresenham import bresenham

debug = True
coloredlogs.install(level=logging.INFO)

if debug:
    logging.debug(">> Debug params active")
    TRIALS = 2
    TIMEOUT_S = 60
    MAX_AGENTS = 51
else:
    TRIALS = 10
    TIMEOUT_S = 600  # 10 min
    MAX_AGENTS = 200

SUCCESSFUL = "successful"
COMPUTATION_TIME = "computation_time"
COST = "cost"

WIDTH = 10000
PATH_ILP = "~/src/optimal-mrppg-journal"  # github.com/ct2034/optimal-mrppg-journal
PATH_ECBS = "~/src/libMultiRobotPlanning"  # github.com/ct2034/libMultiRobotPlanning


@unique
class Planner(Enum):
    RCBS = 0
    ECBS = 1
    ILP = 2


@unique
class Graph(Enum):
    ODRM = 0
    UDRM = 1
    GRID = 2


class TimeoutException(Exception):
    def __init__(self, msg):
        pass


def timeout_handler(signum, frame):
    logging.warn("timeout over")
    raise TimeoutException("timeout over")


def evaluate(fname):
    assert is_result_file(fname), "Please call with result file (*.pkl)"
    fname_graph_adjlist, fname_graph_pos = get_graph_csvs(fname)
    assert os.path.exists(fname_graph_adjlist) and os.path.exists(
        fname_graph_pos), "Please make csv files first `script/write_graph.py csv res/...pkl`"
    fname_graph_undir_adjlist = 
    assert os.path.exists()
    fname_map = resolve_mapname(fname)
    fname_results = get_basename_wo_extension(fname) + ".eval_cen.pkl"

    # read file
    with open(fname, "rb") as f:
        assert is_result_file(fname), "Please call with result file"
        store = pickle.load(f)
    posar = store['posar']
    N = posar.shape[0]
    edgew = store['edgew']
    im = imageio.imread(fname_map)
    _, graph, pos = graphs_from_posar(N, posar)
    make_edges(N, _, graph, posar, edgew, im)
    graph_undir = graph.to_undirected(as_view=True)

    # grid map
    g_grid, posar_grid = make_gridmap(N, im)

    results = {}
    results[SUCCESSFUL] = {}
    results[COMPUTATION_TIME] = {}
    results[COST] = {}

    for i_c, (planner_type, graph_type) in enumerate(itertools.product(Planner, Graph)):
        combination_name = "{}-{}".format(planner_type.name, graph_type.name)
        logging.info("- Combination {}/{}: {}".format(i_c + 1, len(Planner) * len(Graph),
                                                      combination_name))
        results[SUCCESSFUL][combination_name] = {}
        results[COMPUTATION_TIME][combination_name] = {}
        results[COST][combination_name] = {}
        for n_agents in range(25, MAX_AGENTS, 25):
            logging.info("-- n_agents: " + str(n_agents))
            results[SUCCESSFUL][combination_name][str(n_agents)] = []
            results[COMPUTATION_TIME][combination_name][str(n_agents)] = []
            results[COST][combination_name][str(n_agents)] = []
            for i_trial in range(TRIALS):
                logging.info("--= trial {}/{}".format(i_trial + 1, TRIALS))
                if graph_type is Graph.ODRM:
                    g = graph
                    p = posar
                elif graph_type is Graph.UDRM:
                    g = graph_undir
                    p = posar
                elif graph_type is Graph.GRID:
                    g = g_grid
                    p = posar_grid
                random.seed(i_trial)
                succesful, comp_time, cost = plan(N, planner_type, graph_type, n_agents, g, p)
                results[SUCCESSFUL][combination_name][str(n_agents)].append(succesful)
                results[COMPUTATION_TIME][combination_name][str(n_agents)].append(comp_time)
                results[COST][combination_name][str(n_agents)].append(cost)

    logging.info(pretty(results))
    with open(fname_results, 'wb') as f_res:
        pickle.dump(results, f_res)
        logging.info("Written results to: " + fname_results)


def plan(N, planner_type, graph_type, n_agents, g, posar):
    batch = get_unique_batch(N, n_agents)
    start_time = time.time()
    if planner_type is Planner.RCBS:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(TIMEOUT_S)
        try:
            cost, paths = eval_disc(batch, g,
                                    posar, 0, .1)
        except TimeoutException:
            return False, float(TIMEOUT_S), 0
    elif planner_type is Planner.ECBS:
        logging.warn("to be implemented")
        cost = 99
    elif planner_type is Planner.ILP:
        logging.warn("to be implemented")
        cost = 89
    t = time.time() - start_time
    return True, t, cost


def make_gridmap(N, im):
    N_grid = 0
    side_len = im.shape[0]
    edge_len = int(side_len / sqrt(N))
    while N_grid < N:
        edge_len -= 2
        g_grid = nx.Graph()  # undirected
        posar_grid = []
        for i_x, x in enumerate(np.arange(0, side_len, edge_len)):
            for i_y, y in enumerate(np.arange(0, side_len, edge_len)):
                pos = [x, y]
                if is_pixel_free(im, pos):
                    g_grid.add_node(len(posar_grid))
                    posar_grid.append(pos)
        for i_a, i_b in itertools.product(range(len(posar_grid)), repeat=2):
            a = posar_grid[i_a]
            b = posar_grid[i_b]
            if (a[0] == b[0] and isclose(a[1] - b[1], edge_len) or  # up
                    isclose(a[0] - b[0], edge_len) and a[1] == b[1]):  # right
                line = bresenham(
                    int(a[0]),
                    int(a[1]),
                    int(b[0]),
                    int(b[1])
                )
                if all([is_pixel_free(im, x) for x in line]):
                    g_grid.add_edge(i_a, i_b)
        for n in g_grid.nodes:
            if 0 == nx.degree(g_grid, n):
                g_grid.remove_node(n)
        N_grid = len(g_grid.nodes)
        logging.debug(edge_len)
        logging.debug(N_grid)
    return g_grid, posar_grid


def n_to_xy(n):
    return (
        n % WIDTH,
        (n - n % WIDTH) / WIDTH
    )


def xy_to_n(x, y):
    return y * WIDTH + x


def isclose(a, b):
    return abs(a - b) < 1E-9


def pretty(d, indent=0):
    str_out = ''
    for key, value in d.items():
        str_out += ('\t' * indent + str(key)) + '\n'
        if isinstance(value, dict):
            str_out += pretty(value, indent + 1) + '\n'
        else:
            str_out += ('\t' * (indent + 1) + str(value) + '\n')
    return str_out


if __name__ == '__main__':
    fname = sys.argv[1]
    evaluate(fname)
