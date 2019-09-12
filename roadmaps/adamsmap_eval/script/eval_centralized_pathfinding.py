#!/usr/bin/env python2
import itertools
import logging
import os
import pickle
import random
import sys
from enum import Enum, unique
from math import sqrt

import imageio
import networkx as nx
import numpy as np
from adamsmap.adamsmap import is_pixel_free
from adamsmap_eval.filename_verification import (
    get_graph_csvs,
    is_result_file,
    resolve_mapname
)
from adamsmap_eval.eval_disc import (
    get_unique_batch,
    write_csv,
    eval_disc
)
from bresenham import bresenham

TRIALS = 10
TIMEOUT_S = 600  # 10 min

SUCCESS_RATE = "success_rate"
COMPUTATION_TIME = "computation_time"
COST = "cost"

WIDTH = 10000
PATH_ILP = "~/src/optimal-mrppg-journal"  # github.com/ct2034/optimal-mrppg-journal
PATH_ECBS = "~/src/libMultiRobotPlanning"  # github.com/ct2034/libMultiRobotPlanning

logging.basicConfig(level=logging.INFO)


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


def evaluate(fname):
    assert is_result_file(fname), "Please call with result file (*.pkl)"
    fname_graph_adjlist, fname_graph_pos = get_graph_csvs(fname)
    assert os.path.exists(fname_graph_adjlist) and os.path.exists(
        fname_graph_pos), "Please make csv files first `script/write_graph.py csv res/...pkl`"
    fname_map = resolve_mapname(fname)

    # read file
    with open(fname, "rb") as f:
        assert is_result_file(fname), "Please call with result file"
        store = pickle.load(f)
    posar = store['posar']
    N = posar.shape[0]
    edgew = store['edgew']
    im = imageio.imread(fname_map)

    # grid map
    g_grid, posar_grid = make_gridmap(N, im)

    results = {}
    results[SUCCESS_RATE] = {}
    results[COMPUTATION_TIME] = {}
    results[COST] = {}

    for i_c, (planner_type, graph_type) in enumerate(itertools.product(Planner, Graph)):
        combination_name = "{}-{}".format(planner_type.name, graph_type.name)
        logging.info("Combination {}/{}: {}".format(i_c + 1, len(Planner) * len(Graph),
                                                       combination_name))
        results[SUCCESS_RATE][combination_name] = {}
        results[COMPUTATION_TIME][combination_name] = {}
        results[COST][combination_name] = {}
        for n_agents in range(25, 175, 25):
            results[SUCCESS_RATE][combination_name][str(n_agents)] = []
            results[COMPUTATION_TIME][combination_name][str(n_agents)] = []
            results[COST][combination_name][str(n_agents)] = []
            for i_trial in range(TRIALS):
                random.seed(i_trial)
                succ_r, time, cost = plan(planner_type, graph_type, n_agents)
                results[SUCCESS_RATE][combination_name][str(n_agents)].append(succ_r)
                results[COMPUTATION_TIME][combination_name][str(n_agents)].append(time)
                results[COST][combination_name][str(n_agents)].append(cost)

def plan(planner_type, graph_type, n_agents):
    batch = get_unique_batch(N, n_agents)
    if planner_type is Planner.RCBS:
        eval_disc(batch, ge,
                  posar, agent_diameter, v)


def make_gridmap(N, im):
    N_grid = 0
    side_len = im.shape[0]
    edge_len = int(side_len / sqrt(N))
    while N_grid < N:
        edge_len -= 2
        g_grid = nx.Graph()
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

if __name__ == '__main__':
    fname = sys.argv[1]
    evaluate(fname)
