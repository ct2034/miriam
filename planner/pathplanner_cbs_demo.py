import datetime

import numpy as np
import matplotlib.pyplot as plt

from planner.tcbs.plan import plan as plan_cbsext
from planner.eval.display import plot_results

grid = np.zeros([10, 10, 51])
grid[4, 1:9, :] = -1


def pathplan(agent_pos, jobs):
    start_time = datetime.datetime.now()
    # This misuses the cbsext planner as cbs only planner by fixing the assignment
    res_agent_job, res_agent_idle, res_paths = plan_cbsext(agent_pos, jobs, [], [], grid,
                                                           plot=False, filename='pathplanning_only.pkl',
                                                           pathplanning_only_assignment=[(0,), (1,), (2,)])
    print("computation time:", (datetime.datetime.now() - start_time).total_seconds(), "s")
    return res_paths


# input 1
agent_pos = [(1, 1), (2, 1), (3, 1)]  # three agents
jobs = [((1, 7), (9, 1), 0),
        ((1, 8), (8, 1), 0),
        ((9, 8), (4, 2), 0)]  # three jobs

paths = pathplan(agent_pos, jobs)
for p in paths:
    print(p)
plot_results([], paths, [], agent_pos, plt.figure(), grid, [], jobs)
