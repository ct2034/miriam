import os

import numpy as np
from planner.cbs_ext.plan import plot_results
import matplotlib.pyplot as plt

from planner.cbs_ext.plan import generate_config
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
    # assert res_agent_job, "No result" TODO: check!
    assert res_paths, "No result"


if __name__ == "__main__":
    test_cobra_simple()
