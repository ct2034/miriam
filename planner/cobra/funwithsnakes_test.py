import os

import numpy as np

from planner.cobra.funwithsnakes import read_file


def test_read_file(fname='planner/cobra/test.path', plot=False):
    print("cwd: " + str(os.getcwd()))
    grid = np.zeros([10, 10, 100])
    paths = read_file(fname, grid)
    if plot:
        import matplotlib.pyplot as plt
        from planner.cbs_ext.plan import plot_results
        plot_results([], paths, [], [], plt.figure(), grid, [], [])
    assert len(paths) == 3, "Not all agents have paths"
    assert len(paths[0][0]) == 30, "No full paths"


if __name__ == "__main__":
    test_read_file("/home/cch/src/cobra/COBRA/small_collision.task_tp_path", True)
