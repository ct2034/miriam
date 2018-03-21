import os
import unittest
from functools import reduce

import numpy as np

from planner.eval.display import plot_inputs, plot_results
import matplotlib.pyplot as plt

from planner.tcbs.plan import generate_config
from planner.cbs_ext_test import get_data_random
from planner.cobra.funwithsnakes import read_path_file, plan_cobra


@unittest.skip("They are hard to compare, then")
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


def test_cobra_random(plot=False):
    agent_pos, grid, idle_goals, jobs = get_data_random(seed=1,
                                                        map_res=8,
                                                        map_fill_perc=20,
                                                        agent_n=3,
                                                        job_n=3,
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
    assert not jobs_is, "Not all jobs allocated"
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(121)
        plot_inputs(ax, agent_pos, idle_goals, jobs, grid)
        ax2 = fig.add_subplot(122, projection='3d')
        plot_results(ax2, [], res_paths, res_agent_job, agent_pos, grid, [], jobs)
        plt.show()


if __name__ == "__main__":
    test_cobra_random(True)
