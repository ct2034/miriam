import csv
import hashlib
import logging
import os
import random
import subprocess
import time
import uuid
from itertools import product
from typing import Any, List

import numpy as np
import yaml

import tools
from definitions import FREE, INVALID

logger = logging.getLogger(__name__)

BLOCKS_STR = "blocks"


def to_inputfile(gridmap, starts, goals, fname):
    """
    Example:

    map:
    dimensions: [5, 2]
    obstacles:
        - [0, 1]
        - [1, 1]
        - [3, 1]
        - [4, 1]
    agents:
    - name: agent0
        start: [0, 0]
        potentialGoals:
        - [4, 0]
        - [3, 0]
    - name: agent1
        start: [1, 0]
        potentialGoals:
        - [4, 0]
        - [3, 0]
    """
    starts = np.array(starts)
    # goals = np.array(goals)
    obstacles = []
    for x, y in list(zip(*(np.where(gridmap > FREE)))):
        obstacles.append([x.item(), y.item()])
    data = {
        "map": {"dimensions": list(gridmap.shape), "obstacles": obstacles},
        "agents": list(
            map(
                lambda i_a: {
                    "name": f"agent{i_a}",
                    "start": [starts[i_a, 0].item(), starts[i_a, 1].item()],
                    "potentialGoals": goals,
                },
                range(len(starts)),
            )
        ),
    }
    print("data", data)
    with open(fname, "w") as f:
        yaml.dump(data, f, default_flow_style=True)


def read_outfile(fname):
    with open(fname, "r") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    return data


def call_subprocess(
    fname_infile, fname_outfile, suboptimality, timeout, disappear_at_goal
):
    start_time = time.time()
    out_data = INVALID
    process = None
    t = 0
    cmd = [
        os.path.dirname(__file__) + "/libMultiRobotPlanning/build/cbs_ta",
        "-i",
        fname_infile,
        "-o",
        fname_outfile,
    ]
    logger.info(" ".join(cmd))
    try:
        process = subprocess.Popen(
            cmd,
            cwd=os.path.dirname(__file__) + "/libMultiRobotPlanning",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
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
            time.sleep(0.1)
    except subprocess.CalledProcessError as e:
        logger.warning("CalledProcessError")
        logger.warning(e.output)
    finally:
        if process is not None:
            if process.poll() is None:
                process.kill()
            try:
                os.remove(fname_infile)
            except OSError:
                pass
    if os.path.exists(fname_outfile):
        try:
            out_data = read_outfile(fname_outfile)
            t = time.time() - start_time
        except TypeError as e:
            logger.error("TypeError" + str(e))
        try:
            os.remove(fname_outfile)
        except OSError:
            pass
    logger.debug("out_data: " + str(out_data))
    return out_data


def plan_in_gridmap(
    gridmap: np.ndarray, starts, goals, suboptimality, timeout, disappear_at_goal=False
):
    # solving memoryview: underlying buffer is not C-contiguous
    gridmap = np.asarray(gridmap, order="C")
    uuid_str = str(uuid.uuid1())
    fname_infile = "/tmp/" + uuid_str + "_in.yaml"
    fname_outfile = "/tmp/" + uuid_str + "_out.yaml"
    to_inputfile(gridmap, starts, goals, fname_infile)
    print("fname_infile", fname_infile)
    out_data = call_subprocess(
        fname_infile, fname_outfile, suboptimality, timeout, disappear_at_goal
    )
    return out_data


if __name__ == "__main__":
    # gridmap = np.array([
    #     [0, 0, 1, 0],
    #     [0, 0, 1, 0],
    #     [0, 0, 0, 0],
    #     [0, 0, 1, 0],
    #     [0, 0, 1, 0],
    # ])
    # starts = [
    #     [0, 1],
    #     [4, 1]
    # ]
    # goals = [
    #     [0, 3],
    #     [4, 3]
    # ]
    # res = plan_in_gridmap(gridmap, starts, goals, 1.5, 10, True)
    # print(res)

    gridmap = np.array([[0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 1]])
    starts = [[1, 2], [3, 2], [2, 1]]
    goals = [[3, 0], [2, 2], [0, 0]]
    res = plan_in_gridmap(gridmap, starts, goals, 1.5, 10, True)
    print(res)
