import os

import matplotlib.pyplot as plt
import numpy as np

from planner.cbs_ext.plan import plot_results
from planner.cobra_interface.funwithsnakes import read_file


def test_read_file():
    print("cwd: " + str(os.getcwd()))
    grid = np.zeros([10, 10, 100])
    paths = read_file('planner/cobra_interface/test.path', grid)
    # plot_results([], paths, [], [], plt.figure(), grid, [], [])
    assert len(paths) == 3, "Not all agents have paths"
    assert len(paths[0][0]) == 30, "No full paths"
