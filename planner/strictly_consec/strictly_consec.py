import logging
import numpy as np
from itertools import *

from pyflann import FLANN

from planner.cbs_ext.plan import plan as plan_cbsext

logging.getLogger('pyutilib.component.core.pca').setLevel(logging.INFO)

t = tuple


def manhattan_dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def plan_sc(agent_pos, jobs, grid, filename=None):
    res_agent_job = strictly_consec(agent_pos, jobs)

    _, _, res_paths = plan_cbsext(agent_pos, jobs, [], [], grid,
                                  plot=False,
                                  filename='pathplanning_only.pkl',
                                  pathplanning_only_assignment=res_agent_job)
    return res_agent_job, res_paths


def strictly_consec(agents, tasks):
    N_CLOSEST = 2
    TYPE = "float64"
    agents = np.array(agents, dtype=TYPE)

    free_agents = agents.copy()
    free_tasks = tasks.copy()

    consec = {}
    agent_task = {}

    while len(free_tasks) > 0:
        free_tasks_ends = np.array(list(map(lambda a: a[1], free_tasks)), dtype=TYPE)
        free_tasks_starts = np.array(list(map(lambda a: a[0], free_tasks)), dtype=TYPE)
        possible_starts = np.concatenate([free_agents, free_tasks_ends], axis=0)
        flann = FLANN()
        result, dists = flann.nn(
            possible_starts,
            free_tasks_starts,
            N_CLOSEST,
            algorithm="kmeans",
            branching=32,
            iterations=7,
            checks=16)
        nearest = np.unravel_index(np.argmin(dists), [len(possible_starts), N_CLOSEST])
        i_task_start = result[nearest]
        if nearest[0] >= len(free_agents):  # is a task end
            i_task_end = free_tasks_ends[nearest[0] - len(free_agents)]
            consec[t(i_task_end)] = np.where(tasks == free_tasks[i_task_start])  # after this task comes that
        else:  # an agent
            i_agent = np.where(agents == free_agents[nearest[0]])
            agent_task[t(i_agent)] = np.where(tasks == free_tasks[i_task_start])
            free_agents = np.delete(free_agents, nearest[0], axis=0)
        free_tasks = np.delete(free_tasks, i_task_start, axis=0)

    return agent_task, consec
