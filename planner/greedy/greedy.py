import logging
from functools import reduce

import numpy as np
from pyflann import FLANN

from planner.common import path
from planner.tcbs.plan import load_paths, make_unique
from planner.tcbs.plan import plan as plan_cbsext
from planner.tcbs.plan import save_paths

logging.getLogger("pyutilib.component.core.pca").setLevel(logging.INFO)

t = tuple


def manhattan_dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def plan_greedy(agent_pos, jobs, grid, config):
    filename = config["filename_pathsave"]
    load_paths(filename)
    res_agent_job = strictly_consec(agent_pos, jobs, grid)
    assertjobs = reduce(lambda a, b: a + b, res_agent_job, tuple())
    for ij in range(len(jobs)):
        assert ij in assertjobs
    save_paths(filename)
    print(res_agent_job)
    _, _, res_paths = plan_cbsext(
        agent_pos,
        jobs,
        [],
        [],
        grid,
        plot=False,
        config=config,
        pathplanning_only_assignment=res_agent_job,
    )
    return res_agent_job, res_paths


def get_closest(possible_starts, free_tasks_starts, grid, n):
    flann = FLANN()
    result, dists = flann.nn(
        possible_starts,
        free_tasks_starts,
        n,
        algorithm="kmeans",
        branching=32,
        iterations=7,
        checks=16,
    )
    lengths = []
    nearestss = []
    paths = []
    INF = 2 * np.max(np.max(dists))
    for i in range(n):
        temp_nearest = np.unravel_index(np.argmin(dists), [len(possible_starts), n])
        dists[temp_nearest] = INF
        nearestss.append(temp_nearest)

        temp_i_possible_starts = result[temp_nearest]
        temp_i_free_tasks_start = temp_nearest[0]
        p, _ = path(
            tuple(possible_starts[temp_i_possible_starts]),
            tuple(free_tasks_starts[temp_i_free_tasks_start]),
            grid,
            [],
        )
        if p:
            lengths.append(len(p))
        paths.append(p)
    best_path = np.argmin(lengths)
    nearest = nearestss[best_path]
    i_free_tasks_start = nearest[0]
    i_possible_starts = result[nearest]
    return i_free_tasks_start, i_possible_starts, paths[best_path]


def strictly_consec(agents_list, tasks, grid):
    N_CLOSEST = 2
    TYPE = "float64"
    agents = np.array(agents_list, dtype=TYPE)
    tasks = make_unique(tasks)

    free_agents = agents.copy()
    free_tasks = tasks.copy()

    consec = {}
    agent_task_d = {}

    while len(free_tasks) > 0:
        free_tasks_ends = np.array(list(map(lambda a: a[1], free_tasks)), dtype=TYPE)
        free_tasks_starts = np.array(list(map(lambda a: a[0], free_tasks)), dtype=TYPE)
        if len(free_tasks) > len(free_agents):
            possible_starts = np.concatenate([free_agents, free_tasks_ends], axis=0)
        else:
            possible_starts = free_agents
        if len(possible_starts) > 1:
            i_free_tasks_start, i_possible_starts, p = get_closest(
                possible_starts, free_tasks_starts, grid, N_CLOSEST
            )
        else:  # only one start left
            i_free_tasks_start = 0
            i_possible_starts = 0
        if i_possible_starts >= len(free_agents):  # is a task end
            i_task_end = i_possible_starts - len(free_agents)
            consec[i_task_end] = tasks.index(
                free_tasks[i_free_tasks_start]
            )  # after this task comes that
        else:  # an agent
            i_agent = agents_list.index(t(free_agents[i_possible_starts]))
            agent_task_d[i_agent] = tasks.index(t(free_tasks[i_free_tasks_start]))
            free_agents = np.delete(free_agents, i_possible_starts, axis=0)
        free_tasks.pop(i_free_tasks_start)

    agent_task = [tuple() for _ in range(len(agents_list))]
    for k, v in agent_task_d.items():
        agent_task[k] = (v,)
        t_to_check = v
        while t_to_check in consec.keys():
            consec_t = consec[t_to_check]
            agent_task[k] = agent_task[k] + (consec_t,)
            t_to_check = consec_t

    return agent_task
