#!/usr/bin/env python3
import logging
import os
import subprocess
import time
from random import Random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import yaml
from definitions import INVALID, POS
from roadmaps.var_odrm_torch.var_odrm_torch import (make_graph, read_map,
                                                    sample_points)
from scenarios.visualization import plot_with_paths
from tools import hasher

logger = logging.getLogger(__name__)


def call_subprocess(cmd, timeout):
    start_time = time.time()
    success = False
    t = 0.
    logger.debug(" ".join(cmd))
    process = None
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.path.dirname(__file__)
        )
        while t < timeout:
            t = time.time() - start_time
            if process.poll() is not None:
                logger.debug("process.returncode " + str(process.returncode))
                stdoutdata, stderrdata = process.communicate()
                logger.debug("stdoutdata: " + str(stdoutdata))
                if stderrdata:
                    logger.error("stderrdata: " + str(stderrdata))
                else:
                    success = True
                break
            time.sleep(.1)
        if success:
            logger.debug("runtime: " + str(t))
        else:
            logger.debug("timeout: " + str(t))
    except subprocess.CalledProcessError as e:
        logger.warning("CalledProcessError")
        logger.warning(e.output)
    finally:
        if process is not None:
            process.kill()
    logger.debug("success: " + str(success))
    return success


def write_infile(fname, g, starts, goals):
    data = {}
    data["roadmap"] = {}
    data["roadmap"]["undirected"] = True
    data["roadmap"]["allow_wait_actions"] = True
    data["roadmap"]["vertices"] = {}
    for n in g.nodes:
        data["roadmap"]["vertices"][str(n)] = list(
            map(float, g.nodes[n][POS]))
    data["roadmap"]["edges"] = []
    for e in g.edges:
        data["roadmap"]["edges"].append(
            [str(e[0]), str(e[1])])
    data["agents"] = []
    for i in range(len(starts)):
        data["agents"].append({
            "name": "agent" + str(i),
            "start": str(starts[i]),
            "goal": str(goals[i])
        })
    with open(fname, 'w') as f:
        yaml.dump(data, f, default_flow_style=True)


def read_outfile(fname):
    with open(fname, 'r') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    if data['statistics']['success'] == 0:
        return INVALID
    paths = []
    assert 'schedule' in data.keys()
    schedule = data['schedule']
    n_agents = len(schedule)
    paths = [list() for _ in range(n_agents)]
    for k, v in schedule.items():
        i_a = int(k.replace('agent', ''))
        for pose in v:
            paths[i_a].append((pose['v'], pose['t']))
    return paths


def plan_cbsr(g, starts, goals, radius: float = .01, timeout: float = 60.):
    this_dir = os.path.dirname(__file__)
    tmp_dir = "/tmp/"

    # assertions on starts and goals
    assert len(starts) == len(goals), "starts and goals must have same length"
    assert len(starts) > 0, "there must be at least one start"
    assert len(goals) > 0, "there must be at least one goal"
    assert len(starts) == len(np.unique(starts)), "starts must be unique"
    assert len(goals) == len(np.unique(goals)), "goals must be unique"

    # write infile
    hash = hasher([g, starts, goals])
    fname_infile_to_annotate = f"{tmp_dir}{hash}_infile_to_annotate.yaml"
    fname_infile = f"{tmp_dir}{hash}_infile.yaml"
    fname_outfile = f"{tmp_dir}{hash}_outfile.yaml"
    write_infile(fname_infile_to_annotate, g, starts, goals)

    # call annotate_roadmap
    cmd_ar = [
        "python3",
        this_dir +
        "/libMultiRobotPlanning/tools/annotate_roadmap.py",
        fname_infile_to_annotate,
        fname_infile,
        str(radius)
    ]
    logger.debug("call annotate_roadmap")
    success_ar = call_subprocess(cmd_ar, timeout)
    logger.debug("success_ar: " + str(success_ar))

    if not success_ar:
        return INVALID

    # call cbs_roadmap
    cmd_cbsr = [
        this_dir + "/libMultiRobotPlanning/build/cbs_roadmap",
        "-i", fname_infile,
        "-o", fname_outfile]
    logger.debug("call cbs_roadmap")
    success_cbsr = call_subprocess(cmd_cbsr, timeout)
    logger.debug("success_cbsr: " + str(success_cbsr))
    if not success_cbsr:
        with open(fname_infile, 'r') as f:
            content = "".join(f.readlines())
            logger.debug(f"cbsr failed on: >>{content}<<")

    # check output
    paths = INVALID
    if os.path.isfile(fname_outfile):
        paths = read_outfile(fname_outfile)
        logger.debug("paths: " + str(paths))

    # clean up
    for file in [fname_infile, fname_infile_to_annotate, fname_outfile]:
        if os.path.isfile(file):
            os.remove(file)
    return paths


if __name__ == "__main__":
    map_fname: str = "roadmaps/odrm/odrm_eval/maps/x.png"
    rng = Random(1)
    n = 20
    n_agents = 4

    map_img = read_map(map_fname)
    pos = sample_points(n, map_img, rng)
    g = make_graph(pos, map_img)
    starts = rng.sample(range(n), n_agents)
    goals = rng.sample(range(n), n_agents)
    for i_a in range(n_agents):
        assert nx.has_path(g, starts[i_a], goals[i_a])

    paths = plan_cbsr(g, starts, goals)

    # plot
    if paths is not None:
        plot_with_paths(g, paths)
        plt.show()
