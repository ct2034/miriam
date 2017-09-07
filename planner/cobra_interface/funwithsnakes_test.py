import matplotlib.pyplot as plt
import numpy as np

from planner.cbs_ext.plan import plot_results
from planner.cobra_interface.funwithsnakes import read_file


def test_read_file():
    paths = read_file('/home/cch/src/cobra/COBRA/small_collision.task_tp_path')
    plot_results([], paths, [], [], plt.figure(), np.zeros([10, 10, 100]), [], [])
