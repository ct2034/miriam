import os
import uuid
from itertools import product

import numpy as np
import time

MAP_EXT = ".map"
TASK_EXT = ".task"
PATH_EXT = ".task_tp_path"


def plan_cobra(agent_pos, jobs, grid, config):
    agent_job = []
    cobra_filename_base = str(uuid.uuid1())
    cobra_bin = os.getenv("COBRA_BIN")
    if not cobra_bin:  # if env not set, assuming bin in path
        cobra_bin = "cobra"

    job_endpoints_i, agent_points_i = write_map_file(agent_pos, jobs, grid, cobra_filename_base)
    write_task_file(job_endpoints_i, agent_points_i, cobra_filename_base)

    os.system("pwd")
    res = os.system(" ".join([
        cobra_bin,
        cobra_filename_base + MAP_EXT,
        cobra_filename_base + TASK_EXT
    ]))
    assert res == 0, "Error when calling cobra"
    paths = read_path_file(cobra_filename_base + PATH_EXT, grid)
    clean_up(cobra_filename_base)
    return agent_job, paths


def write_map_file(agent_pos, jobs, grid, cobra_filename_base):
    fname = cobra_filename_base + MAP_EXT

    job_endpoints = set()
    for j in jobs:
        job_endpoints.add(j[0])
        job_endpoints.add(j[1])
    job_endpoints = list(job_endpoints)
    sort_by_lines(job_endpoints)
    job_endpoints_i = []
    for j in jobs:
        job_endpoints_i.append(
            (job_endpoints.index(j[0]),
             job_endpoints.index(j[1]))
        )

    agent_points = set()
    for a in agent_pos:
        agent_points.add(a)
    agent_points = list(agent_points)
    sort_by_lines(agent_points)
    agent_points_i = []
    for a in agent_pos:
        agent_points_i.append(agent_points.index(a))

    with open(fname, 'w') as f:
        f.write(",".join(map(str, grid.shape[0:2])) + "\n")
        f.write(str(len(job_endpoints)) + "\n")
        f.write(str(len(agent_points)) + "\n")
        f.write(str(grid.shape[2]) + "\n")
        for y in range(grid.shape[0]):
            line = ""
            for x in range(grid.shape[1]):
                if grid[x][y][0] != 0:
                    line += "@"
                elif (x, y) in job_endpoints:
                    line += "e"
                elif (x, y) in agent_points:
                    line += "r"
                else:
                    line += "."
            f.write(line + "\n")
    f.close()
    return job_endpoints_i, agent_points_i


def write_task_file(job_endpoints_i, agent_points_i, cobra_filename_base):
    fname = cobra_filename_base + TASK_EXT

    with open(fname, 'w') as f:
        f.write(str(len(job_endpoints_i)) + "\n")
        for jei in job_endpoints_i:
            f.write("\t".join([
                "0",
                str(jei[0]),
                str(jei[1]),
                "0",
                "10\n"
            ]))
    f.close()


def read_path_file(fname, grid):
    height = grid.shape[1]
    paths = []
    agent_path = []
    t = 0
    time.sleep(.5)
    while not os.path.exists(fname):
        time.sleep(.5)
    with open(fname, 'r') as f:
        for line in f:
            nums = line.strip().split("\t")
            if len(nums) == 1:  # a new agent
                n = int(nums[0])
                if agent_path:
                    paths.append((agent_path,))
                    agent_path = []
                    t = 0
            else:  # a new point
                agent_path.append(
                    (int(nums[0]),
                     int(nums[1]),
                     t)
                )
                t += 1
        paths.append((agent_path,))
    f.close()
    return paths


def sort_by_lines(l):
    l = list(map(lambda a: (a[1], a[0]), l))
    l.sort()
    l = list(map(lambda a: (a[1], a[0]), l))


def clean_up(cobra_filename_base):
    try:
        os.remove(cobra_filename_base + MAP_EXT)
        os.remove(cobra_filename_base + TASK_EXT)
        os.remove(cobra_filename_base + PATH_EXT)
    except FileNotFoundError:
        pass
