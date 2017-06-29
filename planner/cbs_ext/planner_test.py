import datetime
import logging
import os
import random

import numpy as np
import png
from functools import reduce

import planner.cbs_ext.plan
from tools import get_system_parameters


def load_map(fname = 'cbs_ext/map.png'):
    r = png.Reader(filename=fname)

    x, y, iter, color = r.read()

    m = np.vstack(map(np.sign, iter))
    m = np.array(m, dtype=np.int8) - 1
    return m


def has_edge_collision(paths):
    edges = {}
    for agent_paths in paths:
        if len(agent_paths) > 1:
            path = reduce(agent_paths)
        else:
            path = agent_paths[0]
        for i in range(len(path) - 1):
            a, b = path[i][:2], path[i + 1][:2]
            edge = (a, b) if a > b else (b, a)
            t = path[i][2]
            if t in edges.keys():
                if edge in edges[t]:
                    return True
                else:
                    edges[t].append(edge)
            else:
                edges[t] = []
                edges[t].append(edge)
    return False


def has_vortex_collision(paths):
    vortexes = {}
    for agent_paths in paths:
        if len(agent_paths) > 1:
            path = reduce(agent_paths)
        else:
            path = agent_paths[0]
        for i in range(len(path)):
            vortex = path[i][:2]
            t = path[i][2]
            if t in vortexes.keys():
                if vortex in vortexes[t]:
                    return True
                else:
                    vortexes[t].append(vortex)
            else:
                vortexes[t] = []
                vortexes[t].append(vortex)
    return False


def get_data_labyrinthian(n=1):
    grid = np.zeros([10 * n, 10 * n, 51 * n])
    grid[4 * n, 2 * n:8 * n, :] = -1
    # input
    agent_pos = [(1 * n, 1 * n), (9 * n, 1 * n), (3 * n, 1 * n)]  # three agents
    jobs = [((1 * n, 6 * n), (9 * n, 6 * n), 0), ((3 * n, 3 * n), (7 * n, 3 * n), 0)]  # two jobs 1,6 -> 9,1, 3,3 -> 7,3
    idle_goals = [((9 * n, 7 * n), (5, .5))]  # one idle goal 9,7 with P~N(5,.5)
    # expected results
    agent_job = ((0,), (), (1,))
    agent_idle = ((), (0,), ())
    return agent_idle, agent_job, agent_pos, grid, idle_goals, jobs


def get_unique_coords(max_x, max_y, reset=False):
    global used_coords
    if reset:
        used_coords = set()
    else:
        max_x -= 1
        max_y -= 1
        c = (random.randint(0, max_x),
             random.randint(0, max_y))
        while (c in used_coords):
            c = (random.randint(0, max_x),
                 random.randint(0, max_y))
        assert c not in used_coords
        used_coords.add(c)
        return c


def get_data_random(map_res=10, map_fill_perc=20, agent_n=4, job_n=4, idle_goals_n=2):
    get_unique_coords(None, None, True)  # just to reset ..
    grid = np.zeros([map_res, map_res, map_res ** 2])

    # Fill the map
    for i in range(int(np.floor(map_fill_perc / 100 * map_res ** 2))):
        c = get_unique_coords(map_res, map_res)
        grid[c[1], c[0], :] = -1

    # agents
    agent_pos = []
    for i in range(agent_n):
        agent_pos.append(get_unique_coords(map_res, map_res))

    # jobs
    jobs = []
    for i in range(job_n):
        jobs.append((get_unique_coords(map_res, map_res),
                     get_unique_coords(map_res, map_res),
                     random.randint(0, 4)))

    idle_goals = []
    for i in range(idle_goals_n):
        idle_goals.append((get_unique_coords(map_res, map_res),
                           (random.randint(0, 4),
                            random.randint(1, 20) / 10))
                          )

    return agent_pos, grid, idle_goals, jobs


def test_basic():
    agent_idle, agent_job, agent_pos, grid, idle_goals, jobs = get_data_labyrinthian()

    start_time = datetime.datetime.now()

    res_agent_job, res_agent_idle, res_paths = planner.cbs_ext.plan.plan(agent_pos, jobs, [], idle_goals, grid, filename='')

    print("computation time:", (datetime.datetime.now() - start_time).total_seconds(), "s")

    assert res_agent_job == agent_job, "wrong agent -> job assignment"
    assert res_agent_idle == agent_idle, "wrong agent -> idle_goal assignment"


def test_rand():
    for i in range(5):
        print("\nTEST", i)
        agent_pos, grid, idle_goals, jobs = get_data_random(10, 5, 3, 3, 5)

        start_time = datetime.datetime.now()

        res_agent_job, res_agent_idle, res_paths = planner.cbs_ext.plan.plan(agent_pos, jobs, [], idle_goals, grid, filename='')

        print("computation time:", (datetime.datetime.now() - start_time).total_seconds(), "s")
        print("RESULTS:\nres_agent_job", res_agent_job)
        print("res_agent_idle", res_agent_idle)
        if res_paths is False:
            logging.warning("NO SOLUTION")
        else:
            print("res_paths", res_paths)


def test_benchmark():
    user, _, _ = get_system_parameters()
    if user == 'travis':
        pass


def test_file():
    fname = "/tmp/test.pkl"
    if os.path.exists(fname):
        os.remove(fname)
    assert not os.path.exists(fname), "File exists already"

    agent_idle, agent_job, agent_pos, grid, idle_goals, jobs = get_data_labyrinthian(2)
    start_time = datetime.datetime.now()
    planner.cbs_ext.plan.plan(agent_pos, jobs, [], idle_goals, grid, filename=fname)
    time1 = (datetime.datetime.now() - start_time).total_seconds()
    assert os.path.isfile(fname), "Algorithm has not created a file"

    start_time = datetime.datetime.now()
    planner.cbs_ext.plan.plan(agent_pos, jobs, [], idle_goals, grid, filename=fname)
    time2 = (datetime.datetime.now() - start_time).total_seconds()
    try:
        assert time2 < time1, "It was not faster to work with saved data"
        print("Saving path_save saved us", 100 * (time1 - time2) / time1, "% of time")
    finally:
        os.remove(fname)
        assert not os.path.exists(fname), "File exists after delete"


def test_collision():
    grid = np.zeros([10, 10, 50])
    grid[2:8, 0:4, :] = -1
    grid[2:8, 5:10, :] = -1
    agent_pos = [(3, 1), (5, 1)]
    idle_goals = [((3, 9), (8, .1)), ((5, 9), (8, .1))]

    res_agent_job, res_agent_idle, res_paths = planner.cbs_ext.plan.plan(agent_pos, [], [], idle_goals, grid, filename='',
                                                                         plot=False)
    assert np.array(map(lambda x: len(x) == 0, res_agent_job)).all(), "We don't have to assign jobs"

    assert not has_vortex_collision(res_paths), "There are collisions in vortexes!"
    assert not has_edge_collision(res_paths), "There are collisions in edges!"

def test_consecutive_jobs():
    grid = np.zeros([10, 10, 50])
    agent_pos = [(1, 1)]
    idle_goals = [((3, 9), (8, .1)), ((5, 9), (8, .1))]
    jobs = [((2, 0), (2, 9), -6), ((7, 3), (3, 3), -1.5), ((3, 4), (5, 1), 0)]

    res_agent_job, res_agent_idle, res_paths = planner.cbs_ext.plan.plan(agent_pos=agent_pos,
                                                                         jobs=jobs,
                                                                         alloc_jobs=[],
                                                                         idle_goals=idle_goals,
                                                                         grid=grid,
                                                                         filename='',
                                                                         plot=False)

    assert len(res_agent_idle[0]) == 0, "We don't have to assign idle goals"
    assert len(res_agent_job) == 1, "Not one assigned job"
    assert len(res_agent_job[0]) == 3, "Not all jobs assigned to first agent"
    assert len(res_paths) == 1, "Not one path sets for the agent"
    assert len(res_paths[0]) == 6, "Not six paths for the agent"  # being six due to the oaths to start


def test_same_jobs(plot=False):
    grid = np.zeros([10, 10, 50])
    agent_pos = [(4, 3), (4, 4)]
    idle_goals = [((3, 9), (8, .1)), ((5, 9), (8, .1))]
    jobs = [((0, 0), (9, 9), 0.358605), ((0, 0), (9, 9), 0.002422)]

    res_agent_job, res_agent_idle, res_paths = planner.cbs_ext.plan.plan(agent_pos=agent_pos,
                                                                         jobs=jobs,
                                                                         alloc_jobs=[],
                                                                         idle_goals=idle_goals,
                                                                         grid=grid,
                                                                         filename='',
                                                                         plot=plot)

    assert len(res_agent_idle[0]) == 0, "We don't have to assign idle goals"
    assert len(res_agent_idle[1]) == 0, "We don't have to assign idle goals"
    assert len(res_agent_job) == 2, "Not both jobs assigned"
    assert len(res_agent_job[0]) == 1, "Not all jobs assigned to one agent"
    assert len(res_agent_job[1]) == 1, "Not all jobs assigned to one agent"
    assert len(res_paths) == 2, "Not two path sets for the agents"


def test_concat_paths():
    path1 = [(1, 1, 1), (1, 2, 2), (1, 3, 3)]
    path2 = [(1, 3, 0), (2, 3, 1), (3, 3, 2)]
    res_path = [(1, 1, 1), (1, 2, 2), (1, 3, 3), (2, 3, 4), (3, 3, 5)]

    assert res_path == planner.cbs_ext.plan.concat_paths(path1, path2), "Wrong merging"


def test_idle_goals(plot=False):
    grid = np.zeros([10, 10, 50])
    agent_pos = [(3, 8), (0, 1)]
    idle_goals = [((3, 9), (2, 4)), ]
    jobs = [((0, 0), (9, 9), 1), ]

    res_agent_job, res_agent_idle, res_paths = planner.cbs_ext.plan.plan(agent_pos=agent_pos,
                                                                         jobs=jobs,
                                                                         alloc_jobs=[],
                                                                         idle_goals=idle_goals,
                                                                         grid=grid,
                                                                         filename='',
                                                                         plot=plot)

    assert res_agent_job == ((), (0,))
    assert res_agent_idle == ((0,), ())


def test_vertexswap():
    from planner.cbs_ext.planner_demo_vertexswap import values_vertexswap
    grid, agent_pos, jobs, _, alloc_jobs, start_time = values_vertexswap()

    (_,
     _,
     res_paths) = planner.cbs_ext.plan.plan(agent_pos,
                                            jobs,
                                            alloc_jobs,
                                            [],
                                            grid,
                                            plot=False,
                                            filename='')

    assert not has_vortex_collision(res_paths), "There are collisions in vortexes!"
    assert not has_edge_collision(res_paths), "There are collisions in edges!"


def test_timeshift_path():
    assert [(1, 2, 2), (2, 2, 3)] == planner.cbs_ext.plan.time_shift_path([(1, 2, 0), (2, 2, 1)], 2), "Wrong shifting"


def test_get_nearest():
    assert (1, 1) == planner.cbs_ext.plan.get_nearest([(1, 0), (1, 1), (1, 2)], (0, 1))


if __name__ == "__main__":
    test_vertexswap()
