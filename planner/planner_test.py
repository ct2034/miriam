import datetime
import os

import numpy as np

import planner.plan


def test_basic():
    agent_idle, agent_job, agent_pos, grid, idle_goals, jobs = get_data()

    start_time = datetime.datetime.now()

    res_agent_job, res_agent_idle, res_paths = planner.plan.plan(agent_pos, jobs, [], idle_goals, grid, filename='')

    print("computation time:", (datetime.datetime.now() - start_time).total_seconds(), "s")

    assert res_agent_job == agent_job, "wrong agent -> job assignment"
    assert res_agent_idle == agent_idle, "wrong agent -> idle_goal assignment"


def get_data(n=1):
    grid = np.zeros([10 * n, 10 * n, 51 * n])
    grid[4 * n, 2 * n:8 * n, :] = -1
    # input
    agent_pos = [(1 * n, 1 * n), (9 * n, 1 * n), (3 * n, 1 * n)]  # three agents
    jobs = [((1 * n, 6 * n), (9 * n, 6 * n)), ((3 * n, 3 * n), (7 * n, 3 * n))]  # two jobs 1,6 -> 9,1, 3,3 -> 7,3
    idle_goals = [((9 * n, 7 * n), (5, .5))]  # one idle goal 9,7 with P~N(5,.5)
    # expected results
    agent_job = ((0, 0), (2, 1))
    agent_idle = ((1, 0),)
    return agent_idle, agent_job, agent_pos, grid, idle_goals, jobs


def test_file():
    fname = "/tmp/test.pkl"
    if os.path.exists(fname):
        os.remove(fname)
    assert not os.path.exists(fname), "File exists already"

    agent_idle, agent_job, agent_pos, grid, idle_goals, jobs = get_data(2)
    start_time = datetime.datetime.now()
    planner.plan.plan(agent_pos, jobs, [], idle_goals, grid, filename=fname)
    time1 = (datetime.datetime.now() - start_time).total_seconds()
    assert os.path.isfile(fname), "Algorithm has not created a file"

    start_time = datetime.datetime.now()
    planner.plan.plan(agent_pos, jobs, [], idle_goals, grid, filename=fname)
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

    res_agent_job, res_agent_idle, res_paths = planner.plan.plan(agent_pos, [], [], idle_goals, grid, filename='',
                                                                 plot=False)
    assert len(res_agent_job) == 0, "We don't have to assign jobs"

    ps = res_paths[0] + res_paths[1]
    pss = []
    for m in map(hash, ps):
        pss.append(m)
    pss = np.array(pss)
    unique = np.unique(pss)
    assert len(pss) == len(unique), "Collisions in path_save"


def test_consecutive_jobs():
    grid = np.zeros([10, 10, 50])
    agent_pos = [(1, 1)]
    idle_goals = [((3, 9), (8, .1)), ((5, 9), (8, .1))]
    jobs = [((2, 0), (2, 9), -6), ((7, 3), (3, 3), -1.5), ((3, 4), (5, 1), 0)]

    res_agent_job, res_agent_idle, res_paths = planner.plan.plan(agent_pos=agent_pos,
                                                                 jobs=jobs,
                                                                 alloc_jobs=[],
                                                                 idle_goals=idle_goals,
                                                                 grid=grid,
                                                                 filename='',
                                                                 plot=False)

    assert len(res_agent_idle) == 0, "We don't have to assign idle goals"
    assert len(res_agent_job) == 1, "Not one assigned job"
    assert len(res_agent_job[0][1]) == 3, "Not all jobs assigned to one agent"
    assert len(res_paths) == 1, "Not one path sets for the agent"
    assert len(res_paths[0]) == 3, "Not three paths for the agent"


def test_concat_paths():
    path1 = [(1, 1, 0), (1, 2, 1), (1, 3, 2)]
    path2 = [(1, 3, 0), (2, 3, 1), (3, 3, 2)]
    res_path = [(1, 1, 0), (1, 2, 1), (1, 3, 2), (2, 3, 3), (3, 3, 4)]

    assert res_path == planner.plan.concat_paths(path1, path2), "Wrong merging"
