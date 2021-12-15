#!/usr/bin/env python3
import logging
import os
import subprocess
import time
from random import Random

import matplotlib.pyplot as plt
import networkx as nx
import yaml
from definitions import INVALID, POS
from roadmaps.var_odrm_torch.var_odrm_torch import (make_graph, read_map,
                                                    sample_points)
from scenarios.visualization import plot_with_paths

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def call_subprocess(cmd, timeout):
    start_time = time.time()
    success = False
    t = 0.
    logger.info(" ".join(cmd))
    process = None
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=this_dir
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
    def num2ch(number):
        # ch = ord('A') + number
        # assert ch <= ord('Z')
        return str(number)

    data = {}
    data["roadmap"] = {}
    data["roadmap"]["undirected"] = True
    data["roadmap"]["allow_wait_actions"] = True
    data["roadmap"]["vertices"] = {}
    for n in g.nodes:
        data["roadmap"]["vertices"][num2ch(n)] = list(
            map(float, g.nodes[n][POS]))
    data["roadmap"]["edges"] = []
    for e in g.edges:
        data["roadmap"]["edges"].append(
            [num2ch(e[0]), num2ch(e[1])])
    data["agents"] = []
    for i in range(len(starts)):
        data["agents"].append({
            "name": "agent" + str(i),
            "start": num2ch(starts[i]),
            "goal": num2ch(goals[i])
        })
    with open(fname, 'w') as f:
        yaml.dump(data, f, default_flow_style=True)


def read_outfile(fname):
    with open(fname, 'r') as f:
        data = yaml.load(f)
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


def plan_cbsr(g, starts, goals):
    this_dir = os.path.dirname(__file__)
    fname_infile_to_annotate = this_dir + "/demo_infile_to_annotate.yaml"
    fname_infile = this_dir + "/demo_infile.yaml"
    fname_outfile = this_dir + "/demo_outfile.yaml"
    write_infile(fname_infile_to_annotate, g, starts, goals)

    # call annotate_roadmap
    cmd_ar = [
        "python3",
        this_dir +
        "/libMultiRobotPlanning/tools/annotate_roadmap.py",
        fname_infile_to_annotate,
        fname_infile,
        "0.01"  # radius
    ]
    print("call annotate_roadmap")
    success_ar = call_subprocess(cmd_ar, 60)
    print("success_ar: " + str(success_ar))

    # call cbs_roadmap
    cmd_cbsr = [
        this_dir + "/libMultiRobotPlanning/build/cbs_roadmap",
        "-i", fname_infile,
        "-o", fname_outfile]
    print("call cbs_roadmap")
    success_cbsr = call_subprocess(cmd_cbsr, 60)
    print("success_cbsr: " + str(success_cbsr))

    # check output
    paths = INVALID
    if os.path.isfile(fname_outfile):
        paths = read_outfile(fname_outfile)
        print("paths: " + str(paths))

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
