import logging
import numpy as np

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


def strictly_consec(agents_list, tasks):
    N_CLOSEST = 2
    TYPE = "float64"
    agents = np.array(agents_list, dtype=TYPE)

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
            i_free_tasks_start = nearest[0]
            i_possible_starts = result[nearest]
        else:  # only one start left
            i_free_tasks_start = 0
            i_possible_starts = 0
        if i_possible_starts >= len(free_agents):  # is a task end
            i_task_end = i_possible_starts - len(free_agents)
            consec[i_task_end] = tasks.index(free_tasks[i_free_tasks_start])  # after this task comes that
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
