#!/usr/bin/env python2
import csv
import logging
import os
import random
import re
import subprocess
import sys
import time
from itertools import dropwhile
from math import sqrt
from typing import Any, Tuple

import psutil
import yaml
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)

GRAPH_AL_FNAME = "graph_adjlist.csv"
GRAPH_AL_UNDIR_FNAME = "graph_adjlist_undir.csv"
GRAPH_NP_FNAME = "graph_pos.csv"
INIT_JOBS_FNAME = "init_jobs.csv"
SUBOPTIMALITY = 1.7
TIMEOUT_S = 120  # 2min
MAX_COST = 9999


def max_vertex():
    max_so_far = 0
    with open(GRAPH_AL_FNAME, "r") as f:
        graphreader = csv.reader(f, delimiter=' ')
        for line in graphreader:
            if not line[0].startswith("#"):
                max_so_far = max(
                    max_so_far,
                    max(map(int, line))
                )
    return max_so_far


def get_unique(path):
    assert path.__class__ == list, "must be list"
    assert path[0].__class__ == list, "must be listof lists"
    list_length = len(path)
    last_vertex = path[-1][1]
    unique_reversed = list(dropwhile(
        lambda x: x[1] == last_vertex,
        reversed(path)))
    unique = (list(reversed(unique_reversed))
              + [[len(unique_reversed), last_vertex]])
    assert len(unique) <= list_length
    return unique, len(unique)


def create_initial_jobs_file(N, n_jobs):
    assert n_jobs <= N, "can only have as many jobs as nodes"
    starts_used = set()
    goals_used = set()
    starts = []
    goals = []
    for _ in range(n_jobs):
        ok = False
        while not ok:
            a_start = random.randrange(N)
            a_goal = random.randrange(N)
            if a_start in starts_used or a_goal in goals_used:
                ok = False
            else:
                c, t = plan([a_start], [a_goal],
                            GRAPH_AL_FNAME, GRAPH_NP_FNAME,
                            timeout=TIMEOUT_S, suboptimality=SUBOPTIMALITY)
                if c != MAX_COST:
                    ok = True
                else:
                    ok = False
                    logger.warning(
                        "{} -> {} does not work".format(a_start, a_goal))
        starts.append(a_start)
        starts_used.add(a_start)
        goals.append(a_goal)
        goals_used.add(a_goal)
    with open(INIT_JOBS_FNAME, "w") as f:
        jobswriter = csv.writer(f, delimiter=' ')
        for j in range(n_jobs):
            jobswriter.writerow([starts[j], goals[j]])


def plan(starts, goals, graph_adjlist_fname, graph_pos_fname,
         timeout, suboptimality, remove_outfile=True):
    cwd = os.path.dirname(__file__)+"/libMultiRobotPlanning"
    n_jobs = len(starts)
    assert len(starts) == len(goals), "must have as many starts as goals"
    jobs_fname = "/tmp/%d_jobs.yml" % random.randrange(1E8)
    out_fname = "/tmp/%d_out.yml" % random.randrange(1E8)
    with open(jobs_fname, "w") as f:
        jobswriter = csv.writer(f, delimiter=' ')
        for j in range(n_jobs):
            jobswriter.writerow([starts[j], goals[j]])
    start_time = time.time()
    cost = MAX_COST
    t = 0
    outstr = ""
    cmd = [
        os.path.dirname(__file__) + "/libMultiRobotPlanning/build/ecbs",
        "-a", graph_adjlist_fname,
        "-p", graph_pos_fname,
        "-j", jobs_fname,
        "-o", out_fname,
        "-w", str(suboptimality)]
    logger.info(" ".join(cmd))
    try:
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        while t < timeout:
            t = time.time() - start_time
            if process.poll() is not None:
                logger.info("process.returncode " + str(process.returncode))
                stdoutdata, stderrdata = process.communicate()
                logger.info(stdoutdata)
                if stderrdata:
                    logger.error("stderrdata: " + str(stderrdata))
                outstr = stdoutdata
                break
            time.sleep(.1)
    except subprocess.CalledProcessError as e:
        logger.warning("CalledProcessError")
        logger.warning(e.output)
    finally:
        if process.poll() is None:
            process.kill()
        try:
            os.remove(jobs_fname)
        except OSError:
            pass
    if not os.path.exists(out_fname):
        cost = MAX_COST

    else:
        try:
            cost = get_cost_from_outfile(out_fname)
            t = time.time() - start_time
        except TypeError as e:
            logger.error("TypeError" + str(e))
    logger.debug("cost: " + str(cost))
    if remove_outfile:
        try:
            os.remove(out_fname)
        except OSError:
            pass
    # cleaning up a bit ...
    for proc in psutil.process_iter():
        # check whether the process name matches
        try:
            if proc.name() == "ecbs":
                proc.kill()
                logger.warning("killed it")
        except psutil.NoSuchProcess:
            pass
    return cost, t, out_fname


def get_cost_from_outfile(fname):
    cost = 0
    with open(fname, 'r') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    for agent in data['schedule']:
        last = None
        for node in data['schedule'][agent]:
            if last:
                cost += 1  # dist(node, last)
            last = node
    return cost / len(data['schedule'])


def dist(a, b):
    return sqrt(
        pow(a['x'] - b['x'], 2)
        + pow(a['y'] - b['y'], 2)
    )


def plan_with_n_jobs(n_jobs, N, graph_adjlist_fname):
    random.seed(2034)
    starts = list(range(N))
    goals = list(range(N))
    random.shuffle(starts)
    random.shuffle(goals)
    starts = starts[:n_jobs]
    goals = goals[:n_jobs]
    return plan(starts, goals, graph_adjlist_fname, GRAPH_NP_FNAME,
                timeout=TIMEOUT_S, suboptimality=SUBOPTIMALITY)


def make_undir_graph_file(graph_adjlist_fname, graph_undir_fname):
    def update_graph_dict(d, a, b):
        for (start, end) in [(a, b), (b, a)]:
            if start not in d.keys():
                d[start] = tuple()
            d[start] = d[start] + (end,)
        return d
    with open(graph_adjlist_fname, 'r') as grf:
        grreader = csv.reader(grf, delimiter=' ')
        edges = {}
        for node in grreader:
            if not node[0].startswith("#"):
                for target in node[1:]:
                    edges = update_graph_dict(
                        d=edges,
                        a=int(node[0]),
                        b=int(target)
                    )
    with open(graph_undir_fname, 'w') as gruf:
        grufwriter = csv.writer(gruf, delimiter=' ')
        nodes = list(edges.keys())
        nodes.sort()
        for node in nodes:
            grufwriter.writerow([node] + list(edges[node]))


def write_results(results):
    with open("results.csv", 'w') as f:
        reswriter = csv.writer(f, delimiter=' ')
        for res in results:
            reswriter.writerow(res)


def read_results():
    out = tuple()
    with open("results.csv", 'r') as res:
        resreader = csv.reader(res, delimiter=' ')
        for line in resreader:
            out = out + (list(map(float, line)),)
    return out


if __name__ == '__main__':
    logging.basicConfig()
    logger.setLevel(logging.DEBUG)

    if sys.argv[1] == "eval":
        N = max_vertex()
        # ns = [1, 2, 3, 5, 10, 20, 30, 50, 100]
        # ns = range(1, 20)
        ns = range(10, 190, 10)
        if not os.path.exists(GRAPH_AL_UNDIR_FNAME):
            make_undir_graph_file(GRAPH_AL_FNAME, GRAPH_AL_UNDIR_FNAME)
        results: Tuple[Any, ...] = (ns,)
        for n_jobs in ns:
            cs = []
            ts = []
            for graph_adjlist_fname in [GRAPH_AL_UNDIR_FNAME, GRAPH_AL_FNAME]:
                cost, t = plan_with_n_jobs(n_jobs, N, graph_adjlist_fname, )
                cs.append(cost)
                ts.append(t)
                logger.info(("graph_adjlist_fname: % 24s | n_jobs: %3d |" +
                             " c: % 8.1f | t: % 6.2fs") %
                            (graph_adjlist_fname, n_jobs, cost, t))
            assert len(cs) == 2, "all graphs should have a cost"
            results = results + (cs, ts)
        write_results(results)
    elif sys.argv[1] == "jobsfile":
        create_initial_jobs_file(200, 100)
    elif sys.argv[1] == "plot":
        (ns, cs_u, ts_u, cs_d, ts_d) = read_results()
        x = range(len(ns))
        plt.style.use('bmh')
        fig, (ax_cost, ax_time) = plt.subplots(2, 1)
        fig.tight_layout()
        ax_cost.plot(x, cs_u, label='undirected')
        ax_cost.plot(x, cs_d, label='directed')
        ax_time.plot(x, ts_u, label='undirected')
        ax_time.plot(x, ts_d, label='directed')
        plt.setp(ax_cost, xticks=x, xticklabels=map(str, map(int, ns)),
                 title="Cost")
        plt.setp(ax_time, xticks=x, xticklabels=map(str, map(int, ns)),
                 title="Computation Time")
        ax_cost.legend()
        ax_time.legend()
        plt.show()
    else:
        assert False, "choose either eval or plot"
