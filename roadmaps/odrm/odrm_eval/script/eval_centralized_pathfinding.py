#!/usr/bin/env python2
import csv
import datetime
import itertools
import logging
import numpy as np
import os
import pickle
import psutil
import uuid
import random
import signal
import sys
import time
from enum import Enum
from math import sqrt
from typing import Dict

import benchmark_ecbs
import benchmark_ilp
import coloredlogs
import imageio
import networkx as nx
from odrm.odrm import is_pixel_free
from odrm.eval_disc import get_unique_batch, eval_graph, graphs_from_posar, make_edges
from odrm_eval.filename_verification import (
    get_graph_csvs,
    get_graph_undir_csv,
    is_result_file,
    is_eval_cen_file,
    resolve_mapname,
    get_basename_wo_extension,
)
from bresenham import bresenham

coloredlogs.install(level=logging.INFO)

TRIALS = 5
TIMEOUT_S = 5 * 60  # 5 min
MIN_AGENTS = 25
MAX_AGENTS = 151
STEP_AGENTS = 25

SUCCESSFUL = "successful"
COMPUTATION_TIME = "computation_time"
COST = "cost"

WIDTH = 10000


class Planner(Enum):
    RCBS = 0
    ILP = 1
    ECBS = 2


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
        fname_graph_pos
    ), "Please make csv files first `script/write_graph.py csv res/...pkl`"
    fname_graph_undir_adjlist = get_graph_undir_csv(fname)
    assert os.path.exists(
        fname_graph_undir_adjlist
    ), "Please make csv files first `script/write_graph.py csv res/...pkl`"
    fname_map = resolve_mapname(fname)
    fname_eval_results = (
        "res/"
        + get_basename_wo_extension(fname)
        + ".eval_cen."
        + str(uuid.uuid1())
        + ".pkl"
    )
    assert is_eval_cen_file(fname_eval_results)

    # read file
    with open(fname, "rb") as f:
        assert is_result_file(fname), "Please call with result file"
        store = pickle.load(f)
    posar = store["posar"]
    N = posar.shape[0]
    edgew = store["edgew"]
    im = imageio.imread(fname_map)
    _, graph, pos = graphs_from_posar(N, posar)
    make_edges(N, _, graph, posar, edgew, im)
    graph_undir = graph.to_undirected(as_view=True)

    # grid map
    g_grid, posar_grid, fname_grid_adjlist, fname_grid_posar = make_gridmap(
        N, im, fname
    )
    logging.info("fname_grid_adjlist: " + fname_grid_adjlist)
    logging.info("fname_grid_posar: " + fname_grid_posar)
    logging.info("fname_graph_adjlist: " + fname_graph_adjlist)

    # results
    # type: Dict[str, Dict[str, Dict[str, list]]]
    eval_results = {SUCCESSFUL: {}, COMPUTATION_TIME: {}, COST: {}}
    global_seed = random.randint(0, 1000)

    # the evaluation per combination
    n_agentss = range(MIN_AGENTS, MAX_AGENTS, STEP_AGENTS)
    # planner_iter = Planner
    # planner_iter = [Planner.ILP]
    planner_iter = [Planner.ECBS, Planner.RCBS]

    graph_iter = Graph
    # graph_iter = [Graph.GRID]

    time_estimate = (
        len(planner_iter) * len(graph_iter) * len(n_agentss) * TRIALS * TIMEOUT_S
    )
    logging.info(
        "(worst case) runtime estimate: {} (h:m:s)".format(
            str(datetime.timedelta(seconds=time_estimate))
        )
    )

    for i_c, (planner_type, graph_type) in enumerate(
        itertools.product(planner_iter, graph_iter)
    ):
        combination_name = "{}-{}".format(planner_type.name, graph_type.name)
        logging.info(
            "- Combination {}/{}: {}".format(
                i_c + 1, len(Planner) * len(Graph), combination_name
            )
        )
        eval_results[SUCCESSFUL][combination_name] = {}
        eval_results[COMPUTATION_TIME][combination_name] = {}
        eval_results[COST][combination_name] = {}
        for n_agents in n_agentss:
            logging.info("-- n_agents: " + str(n_agents))
            eval_results[SUCCESSFUL][combination_name][str(n_agents)] = []
            eval_results[COMPUTATION_TIME][combination_name][str(n_agents)] = []
            eval_results[COST][combination_name][str(n_agents)] = []
            for i_trial in range(TRIALS):
                logging.info("--= trial {}/{}".format(i_trial + 1, TRIALS))
                if graph_type is Graph.ODRM:
                    g = graph
                    p = posar
                    fname_adjlist = fname_graph_adjlist
                    fname_posar = fname_graph_pos
                    n = N
                elif graph_type is Graph.UDRM:
                    g = graph_undir
                    p = posar
                    fname_adjlist = fname_graph_undir_adjlist
                    fname_posar = fname_graph_pos
                    n = N
                elif graph_type is Graph.GRID:
                    g = g_grid
                    p = posar_grid
                    fname_adjlist = fname_grid_adjlist
                    fname_posar = fname_grid_posar
                    n = len(posar_grid)
                # this makes cases with more agents comparable to runs with less
                random.seed(i_trial + global_seed)
                run_it = True
                try:
                    # if the run with the last number of agents on this trial failed ...
                    if (
                        eval_results[SUCCESSFUL][combination_name][
                            str(n_agents - STEP_AGENTS)
                        ][i_trial]
                        is False
                    ):
                        run_it = False
                except Exception as e:
                    logging.debug("e {}".format(e))
                    logging.debug("e.message {}".format(e.message))
                    pass
                if run_it:
                    succesful, comp_time, cost = plan(
                        n,
                        planner_type,
                        graph_type,
                        n_agents,
                        g,
                        p,
                        fname_adjlist,
                        fname_posar,
                    )
                else:
                    logging.warn("skipping because last agent number timed out, too")
                    succesful, comp_time, cost = False, TIMEOUT_S, 0
                eval_results[SUCCESSFUL][combination_name][str(n_agents)].append(
                    succesful
                )
                eval_results[COMPUTATION_TIME][combination_name][str(n_agents)].append(
                    comp_time
                )
                eval_results[COST][combination_name][str(n_agents)].append(cost)

                # saving / presenting results
                logging.info("Saving ..")
                logging.info(pretty(eval_results))
                with open(fname_eval_results, "wb") as f_res:
                    pickle.dump(eval_results, f_res)
                    logging.info("Saved results to: " + fname_eval_results)


def plan(n, planner_type, graph_type, n_agents, g, posar, fname_adjlist, fname_posar):
    batch = get_unique_batch(n, n_agents)
    start_time = time.time()
    if planner_type is Planner.RCBS:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(TIMEOUT_S)
        try:
            cost, paths = eval_graph(batch, g, posar)
        except TimeoutException:
            return False, float(TIMEOUT_S), 0
    elif planner_type is Planner.ECBS:
        assert count_processes_with_name("ecbs") < 3
        try:
            cost, _ = benchmark_ecbs.plan(
                starts=batch[:, 0],
                goals=batch[:, 1],
                graph_adjlist_fname=fname_adjlist,
                graph_pos_fname=fname_posar,
                timeout=TIMEOUT_S,
                cwd=os.path.dirname(__file__) + "/../",
            )
        except TimeoutException:
            return False, float(TIMEOUT_S), 0
        if cost == benchmark_ecbs.MAX_COST:
            return False, float(TIMEOUT_S), 0
    elif planner_type is Planner.ILP:
        assert count_processes_with_name("java") < 6
        try:
            paths, _ = benchmark_ilp.plan(
                starts=batch[:, 0],
                goals=batch[:, 1],
                N=n,
                graph_fname=os.path.abspath(os.path.dirname(__file__))
                + "/../"
                + fname_adjlist,
                timeout=TIMEOUT_S,
            )
        except TimeoutException:
            return False, float(TIMEOUT_S), 0
        cost = cost_from_paths(paths, posar) / n_agents
        if len(paths) < n_agents:
            return False, float(TIMEOUT_S), 0
    t = time.time() - start_time
    return True, t, cost


def cost_from_paths(paths, posar):
    cost = 0
    for path in paths:
        prev = None
        for [_, v] in path:
            if prev:
                cost += 1  # dist(posar[prev], posar[v])
            prev = v
    return cost


def make_gridmap(N, im, fname):
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
        for i_n, i_b in itertools.product(range(len(posar_grid)), repeat=2):
            a = posar_grid[i_n]
            b = posar_grid[i_b]
            if (
                a[0] == b[0]
                and isclose(a[1] - b[1], edge_len)  # up
                or isclose(a[0] - b[0], edge_len)
                and a[1] == b[1]
            ):  # right
                line = bresenham(int(a[0]), int(a[1]), int(b[0]), int(b[1]))
                if all([is_pixel_free(im, x) for x in line]):
                    g_grid.add_edge(i_n, i_b)
        for n in g_grid.nodes:
            if 0 == nx.degree(g_grid, n):
                g_grid.remove_node(n)
        N_grid = len(g_grid.nodes)
        logging.debug("edge_len {}".format(edge_len))
        logging.debug("N_grid {}".format(N_grid))

        grid_adjlist_fname = (
            "res/" + get_basename_wo_extension(fname) + ".grid_adjlist.csv"
        )
        grid_posar_fname = "res/" + get_basename_wo_extension(fname) + ".grid_pos.csv"
        g_undir = nx.DiGraph()
        for e in g_grid.edges:
            g_undir.add_edge(e[0], e[1])
            g_undir.add_edge(e[1], e[0])
        nx.write_adjlist(g_undir, grid_adjlist_fname)
        remove_comment_lines(grid_adjlist_fname)
        with open(grid_posar_fname, "w") as f_csv:
            writer = csv.writer(f_csv, delimiter=" ")
            for pos in posar_grid:
                writer.writerow(pos)

    return g_grid, posar_grid, grid_adjlist_fname, grid_posar_fname


def n_to_xy(n):
    return (n % WIDTH, (n - n % WIDTH) / WIDTH)


def xy_to_n(x, y):
    return y * WIDTH + x


def isclose(a, b):
    return abs(a - b) < 1e-9


def pretty(d, indent=0):
    str_out = ""
    for key, value in d.items():
        str_out += ("\t" * indent + str(key)) + "\n"
        if isinstance(value, dict):
            str_out += pretty(value, indent + 1) + "\n"
        else:
            str_out += "\t" * (indent + 1) + str(value) + "\n"
    return str_out


def remove_comment_lines(fname):
    tmp_fname = fname + "TMP"
    os.rename(fname, tmp_fname)
    with open(tmp_fname, "r") as f_tmp:
        with open(fname, "w") as f:
            for line in f_tmp.readlines():
                if not line.startswith("#"):
                    f.write(line)
    os.remove(tmp_fname)


def dist(a, b):
    return sqrt(pow(a[0] - b[0], 2) + pow(a[1] - b[1], 2))


def count_processes_with_name(process_name):
    count = 0
    for proc in psutil.process_iter():
        try:
            if process_name in proc.name():
                count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return count


if __name__ == "__main__":
    fname = sys.argv[1]
    evaluate(fname)
