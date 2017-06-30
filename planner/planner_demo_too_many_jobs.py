import datetime

import numpy as np

from planner.cbs_ext.plan import plan

grid = np.zeros([10, 10, 101])
grid[4, 0:6, :] = -1
grid[8, 4:9, :] = -1

# input
agent_pos = [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5)]
jobs = [((2, 0), (2, 9), 0),
        ((7, 3), (3, 3), 0),
        ((3, 5), (7, 1), 0),
        ((3, 6), (6, 1), 0),
        # ((3, 7), (5, 1), 0),
        ((3, 8), (4, 1), 0)]
idle_goals = [((9, 7), (5, .5)), ((9, 8), (20, .5))]
alloc_jobs = []

start_time = datetime.datetime.now()

(res_agent_job,
 res_agent_idle,
 res_paths) = plan(agent_pos,
                   jobs,
                   alloc_jobs,
                   idle_goals,
                   grid,
                   plot=True,
                   filename='')

print("computation time:",
      (datetime.datetime.now() - start_time).total_seconds(),
      "s")

print(res_agent_job, res_agent_idle, res_paths)
