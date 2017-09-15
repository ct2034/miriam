import os
import unittest
from functools import reduce

import numpy as np

from planner.cbs_ext.plan import plot_results, plot_inputs
import matplotlib.pyplot as plt

from planner.cbs_ext.plan import generate_config
from planner.cbs_ext_test import get_data_random
from planner.cobra.funwithsnakes import read_path_file, plan_cobra


def test_read_map(fname='planner/cobra/test.path', plot=False):
    print("cwd: " + str(os.getcwd()))
    grid = np.zeros([10, 10, 100])
    paths = read_path_file(fname, grid)
    if plot:
        plot_results([], paths, [], [], plt.figure(), grid, [], [])
    assert len(paths) == 3, "Not all agents have paths"
    assert len(paths[0][0]) == 30, "No full paths"


def test_cobra_simple(plot=False):
    grid = np.zeros([5, 5, 30])
    res_agent_job, res_paths = plan_cobra(
        [(1, 1), (2, 2)],
        [((3, 3), (1, 4), 0), ((4, 1), (0, 0), 0)],
        grid,
        generate_config()
    )
    if plot:
        plot_results([], res_paths, [], [], plt.figure(), grid, [], [])
    assert res_agent_job, "No result"
    assert res_paths, "No result"


def test_cobra_random():
    agent_pos, grid, idle_goals, jobs = get_data_random(map_res=8,
                                                        map_fill_perc=20,
                                                        agent_n=5,
                                                        job_n=5,
                                                        idle_goals_n=0)
    res_agent_job, res_paths = plan_cobra(
        agent_pos,
        jobs,
        grid,
        generate_config()
    )
    print(res_agent_job)
    all_alloc = reduce(lambda a, b: a + b, res_agent_job, tuple())
    jobs_is = list(range(len(jobs)))
    for i_j in all_alloc:
        jobs_is.remove(i_j)
    # assert not jobs_is, "Not all joby allocated"
    fig = plot_inputs(agent_pos, idle_goals, jobs, grid, show=False, subplot=121)
    plot_results([], res_paths, [], [], fig, grid, [], jobs)
    plt.show()



if __name__ == "__main__":
    test_cobra_random()
