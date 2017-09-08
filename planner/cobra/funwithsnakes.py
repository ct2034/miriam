import os
import uuid
from itertools import product

import numpy as np

MAP_EXT = ".map"
TASK_EXT = ".map"
PATH_EXT = ".map"


def plan_cobra(agent_pos, jobs, grid, config):
    agent_job = []
    cobra_filename_base = str(uuid.uuid1())
    cobrabin = os.getenv("COBRA_BIN")

    job_endpoints_i, agent_points_i = write_map_file(agent_pos, jobs, grid, cobra_filename_base)

    os.system("pwd")
    res = os.system(" ".join([
        cobrabin,
        cobra_filename_base + MAP_EXT,
        cobra_filename_base + TASK_EXT
    ]))
    assert res == 0, "Error when calling cobra"

    paths = read_path_file(cobra_filename_base + PATH_EXT, grid)
    return agent_job, paths


def write_map_file(agent_pos, jobs, grid, cobra_filename_base):
    fname = cobra_filename_base + MAP_EXT

    job_endpoints = set()
    for j in jobs:
        job_endpoints.add(j[0])
        job_endpoints.add(j[1])
    job_endpoints = list(job_endpoints)
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
    agent_points_i = []
    for a in agent_pos:
        agent_points_i.append(agent_points.index(a))

    with open(fname, 'w') as f:
        f.write(",".join(map(str, grid.shape[0:2])) + "\n")
        f.write(str(len(job_endpoints)) + "\n")
        f.write(str(len(agent_points)) + "\n")
        f.write(str(grid.shape[2]) + "\n")
        for x in range(grid.shape[0]):
            line = ""
            for y in range(grid.shape[1]):
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


def read_path_file(fname, grid):
    height = grid.shape[1]
    paths = []
    agent_path = []
    t = 0
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
                     height - int(nums[1]),
                     t)
                )
                t += 1
        paths.append((agent_path,))
    return paths
