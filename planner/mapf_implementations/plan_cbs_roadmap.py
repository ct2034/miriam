#!/usr/bin/env python3
import logging
import os
import subprocess
from random import Random
from typing import List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import yaml

from definitions import INVALID, POS
from planner.mapf_implementations.libMultiRobotPlanning.tools import annotate_roadmap
from roadmaps.var_odrm_torch.var_odrm_torch import (
    make_graph_and_flann,
    read_map,
    sample_points,
)
from scenarios.visualization import plot_with_paths
from tools import hasher

logger = logging.getLogger(__name__)


def call_subprocess(cmd, timeout):
    success = False
    logger.debug(" ".join(cmd))
    process = None
    try:
        process = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.path.dirname(__file__),
            timeout=timeout,
        )
        logger.debug("returncode " + str(process.returncode))
        logger.debug("stdout: " + str(process.stdout))
        logger.debug("stderr: " + str(process.stderr))
        if process.returncode == 0:
            success = True
    except subprocess.CalledProcessError as e:
        logger.warning("CalledProcessError")
        logger.warning(e.output)
    except subprocess.TimeoutExpired as e:
        logger.debug("TimeoutExpired")
        logger.debug(e.output)
    logger.debug("success: " + str(success))
    return success


def write_roadmap_file(fname: str, g: nx.Graph, radius: float):
    data = {}  # type: ignore
    data["roadmap"] = {}
    data["roadmap"]["undirected"] = True
    data["roadmap"]["allow_wait_actions"] = True
    data["roadmap"]["vertices"] = {}
    for n in g.nodes:
        data["roadmap"]["vertices"][str(n)] = list(map(float, g.nodes[n][POS]))
    data["roadmap"]["edges"] = []
    for e in g.edges:
        data["roadmap"]["edges"].append([str(e[0]), str(e[1])])
    data = annotate_roadmap.add_self_edges(data)
    data = annotate_roadmap.add_edge_conflicts(radius, data)
    with open(fname, "w") as f:
        yaml.dump(data, f, default_flow_style=True)
    return data


def write_infile(fname, data, starts, goals):
    data["agents"] = []
    for i in range(len(starts)):
        data["agents"].append(
            {"name": "agent" + str(i), "start": str(starts[i]), "goal": str(goals[i])}
        )
    with open(fname, "w") as f:
        yaml.dump(data, f, default_flow_style=True)


def read_outfile(fname):
    delete = False
    with open(fname, "r") as f:
        try:
            data = yaml.load(f, Loader=yaml.SafeLoader)
        except yaml.reader.ReaderError:
            # trying to delete probelmatic file
            delete = True
            data = None
    if delete:
        try:
            os.remove(fname)
        except Exception:
            pass
    if data is None:
        return INVALID
    if data["statistics"]["success"] == 0:
        return INVALID
    paths = []
    assert "schedule" in data.keys()
    schedule = data["schedule"]
    n_agents = len(schedule)
    paths = [list() for _ in range(n_agents)]
    for k, v in schedule.items():
        i_a = int(k.replace("agent", ""))
        t = 0
        for pose in v:
            try:
                paths[i_a].append((pose["v"], t))
                t += 1
            except KeyError:
                return INVALID
    return paths


def plan_cbsr(
    g: nx.Graph,
    starts: List[int],
    goals: List[int],
    radius: float,
    timeout: float,
    skip_cache: bool,
    ignore_finished_agents: bool = True,
):
    this_dir = os.path.dirname(__file__)
    cache_dir = os.path.join(this_dir, "cache")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # assertions on starts and goals
    assert len(starts) == len(goals), "starts and goals must have same length"
    assert len(starts) > 0, "there must be at least one start"
    assert len(goals) > 0, "there must be at least one goal"
    if not ignore_finished_agents:
        if not len(starts) == len(np.unique(starts)):  # starts must be unique
            return INVALID
        if not len(goals) == len(np.unique(goals)):  # goals must be unique
            return INVALID

    # make roadmap (with caching)
    hash_roadmap = hasher([g, radius])
    fname_roadmap = f"{cache_dir}/{hash_roadmap}_roadmap.yaml"
    data = None
    if os.path.exists(fname_roadmap) and not skip_cache:
        try:
            with open(fname_roadmap, "r") as f:
                data = yaml.load(f, Loader=yaml.SafeLoader)
        except (yaml.YAMLError, FileNotFoundError) as e:
            logging.warning(e)
            pass
    if not data:  # nothing was loaded
        if os.path.exists(fname_roadmap):
            try:
                os.remove(fname_roadmap)
            except FileNotFoundError:
                pass
        data = write_roadmap_file(fname_roadmap, g, radius)
    if data is None:  # error
        logging.warning("no annotated roadmap")
        return INVALID

    # write infile
    hash = hasher([g, starts, goals, ignore_finished_agents])
    fname_infile = f"{cache_dir}/{hash}_infile.yaml"
    fname_outfile = f"{cache_dir}/{hash}_outfile.yaml"
    if not os.path.exists(fname_infile) or skip_cache:
        write_infile(fname_infile, data, starts, goals)

    # call cbs_roadmap
    if not os.path.exists(fname_outfile) or skip_cache:
        cmd_cbsr = [
            this_dir + "/libMultiRobotPlanning/build/cbs_roadmap",
            "-i",
            fname_infile,
            "-o",
            fname_outfile,
        ]
        if ignore_finished_agents:
            cmd_cbsr += [
                "--disappear-at-goal",
            ]
        logger.debug("call cbs_roadmap")
        success_cbsr = call_subprocess(cmd_cbsr, timeout)
        logger.debug("success_cbsr: " + str(success_cbsr))
        if not success_cbsr:
            try:
                with open(fname_infile, "r") as f:
                    content = "".join(f.readlines())
                    logger.debug(f"cbsr failed on: >>{content}<<")
            except FileNotFoundError:
                pass

    # check output
    paths = INVALID
    if os.path.isfile(fname_outfile):
        paths = read_outfile(fname_outfile)
        logger.debug("paths: " + str(paths))

    return paths


if __name__ == "__main__":
    g = nx.Graph()
    g.add_edges_from([(0, 1), (1, 2), (2, 3)])
    nx.set_node_attributes(g, {0: (0, 0), 1: (1, 0), 2: (2, 0), 3: (3, 0)}, POS)
    starts = [0, 3]
    goals = [2, 1]

    paths = plan_cbsr(g, starts, goals, 0.01, 1, True, ignore_finished_agents=True)

    # plot
    if paths is not INVALID:
        plot_with_paths(g, paths)
        plt.show()

    map_fname: str = "roadmaps/odrm/odrm_eval/maps/x.png"
    rng = Random(1)
    n = 20
    n_agents = 4

    map_img = read_map(map_fname)
    pos = sample_points(n, map_img, rng)
    g, _ = make_graph_and_flann(pos, map_img, n, rng)
    starts = rng.sample(range(n), n_agents)
    goals = rng.sample(range(n), n_agents)
    starts = [0, 0]
    goals = [1, 0]
    for i_a in range(n_agents):
        assert nx.has_path(g, starts[i_a], goals[i_a])

    paths = plan_cbsr(g, starts, goals, 0.01, 30, True, ignore_finished_agents=True)

    # plot
    if paths is not INVALID:
        plot_with_paths(g, paths)
        plt.show()
