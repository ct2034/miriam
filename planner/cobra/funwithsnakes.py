import os
import uuid
from itertools import product

import numpy as np
import time

MAP_EXT = ".map"
TASK_EXT = ".task"
PATH_EXT = ".task_tp_path"
OTHER_PATH_EXT = ".task_tptr_path"

def plan_cobra(agent_pos, jobs, grid, config):
    agent_job = []
    cobra_filename_base = str(uuid.uuid1())
    cobra_bin = os.getenv("COBRA_BIN")
    if not cobra_bin:  # if env not set, assuming bin in path
        cobra_bin = "cobra"

    job_endpoints_i, agent_points_i = write_map_file(agent_pos, jobs, grid, cobra_filename_base)
    write_task_file(job_endpoints_i, agent_points_i, cobra_filename_base)

    time.sleep(1)

    pwd = os.getcwd()
    cmd = " ".join([
        cobra_bin,
        cobra_filename_base + MAP_EXT,
        cobra_filename_base + TASK_EXT
    ])
    res = os.system(cmd)

    time.sleep(.2)

    try:
        if res != 0:
            raise RuntimeError("Error when calling cobra: " + cmd + "\nin: " + pwd)
        paths = read_path_file(cobra_filename_base + PATH_EXT, grid)
        agent_job, paths = allocation_from_paths(paths, agent_pos, jobs)
        paths = make_paths_comparable(paths, agent_job, agent_pos, jobs)
        return agent_job, paths
    finally:
        clean_up(cobra_filename_base)


def write_map_file(agent_pos, jobs, grid, cobra_filename_base):
    fname = cobra_filename_base + MAP_EXT
    grid = np.swapaxes(grid, 0, 1)

    job_endpoints = set()
    for j in jobs:
        job_endpoints.add(j[0])
        job_endpoints.add(j[1])
    job_endpoints = list(job_endpoints)

    job_endpoints_sorted = []

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
                    job_endpoints_sorted.append((x, y))
                elif (x, y) in agent_points:
                    line += "r"
                else:
                    line += "."
            f.write(line + "\n")
    f.close()

    job_endpoints_i = []
    for j in jobs:
        job_endpoints_i.append(
            (job_endpoints_sorted.index(j[0]),
             job_endpoints_sorted.index(j[1]))
        )

    return job_endpoints_i, agent_points_i


def write_task_file(job_endpoints_i, agent_points_i, cobra_filename_base):
    fname = cobra_filename_base + TASK_EXT

    with open(fname, 'w') as f:
        f.write(str(len(job_endpoints_i)) + "\n")
        for jei in job_endpoints_i:
            # jei = job_endpoints_i[0]
            f.write("\t".join([
                "0",
                str(jei[0]),
                str(jei[1]),
                "0",
                "10\n"
            ]))
    f.close()


def read_path_file(fname, grid):
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
    new_paths = []
    for pathset in paths:
        path = pathset[0]
        prev = (-1,-1)
        n_same = 0
        for pose in path:
            if prev == pose[:2]:
                n_same += 1
            else:
                n_same = 0
            prev = pose[:2]
        pathset = (path[0:-n_same], path[-n_same:-1])
        new_paths.append(pathset)    
    f.close()
    return new_paths


def allocation_from_paths(paths, agent_pos, jobs):
    agent_job = []
    starts = {}
    goals = {}
    free_jobs = list(range(len(jobs)))
    for i_j, job in enumerate(jobs):
        start = job[0]
        goal = job[1]
        if start in starts:
            starts[start].append(i_j)
        else:
            starts[start] = [i_j]
        if goal in goals:
            goals[goal].append(i_j)
        else:
            goals[goal] = [i_j]
    new_paths = []
    for i_a, agent in enumerate(agent_pos):
        for path in paths:
            if path[0][0][:2] == agent:
                new_paths.append(path)
    assert len(new_paths) == len(paths), "Got not all paths back"
    paths = new_paths
    for i_a, agent in enumerate(agent_pos):
        agent_starts = []
        agent_goals = []
        agent_alloc = tuple()
        path = paths[i_a][0]
        assert path[0][:2] == agent, "Path must start at agent pos"
        for pose in path:
            pose = pose[:2]
            if pose in starts:
                agent_starts.append(pose)
            if pose in goals:
                agent_goals.append(pose)
        for s, g in product(agent_starts, agent_goals):
            for i_j in starts[s]:
                if i_j in goals[g] and i_j in free_jobs:
                    agent_alloc += (i_j,)
                    free_jobs.remove(i_j)
        agent_job.append(agent_alloc)
        new_paths = []
    return agent_job, paths


def make_paths_comparable(paths, agent_job, agent_pos, jobs):
    paths_out = []
    for i_a in range(len(agent_pos)):
        paths_agent = paths[i_a]
        if len(paths_agent) == 2:
            paths_agent_out = []
            path = paths_agent[0]
            assert path[0][:2] == agent_pos[i_a]
            temp_p = (-1, -1)
            i_p = 0
            while temp_p != jobs[agent_job[i_a][0]][0]:  # start
                temp_p = path[i_p][:2]
                i_p += 1
            paths_agent_out.append(path[0:i_p])
            path_out = []
            for p in path[i_p - 1:]:
                path_out.append(tuple([p[0], p[1], p[2] + 1]))
            paths_agent_out.append(path_out)
            paths_out.append(tuple(paths_agent_out))
        else:
            assert False
    return paths_out




def sort_by_lines(l):
    l = list(map(lambda a: (a[1], a[0]), l))
    l.sort()
    l = list(map(lambda a: (a[1], a[0]), l))


def clean_up(cobra_filename_base):
    try:
        os.remove(cobra_filename_base + MAP_EXT)
        os.remove(cobra_filename_base + TASK_EXT)
        os.remove(cobra_filename_base + PATH_EXT)
        os.remove(cobra_filename_base + OTHER_PATH_EXT)
    except FileNotFoundError:
        pass
