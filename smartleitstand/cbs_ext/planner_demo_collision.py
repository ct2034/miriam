import datetime

import numpy as np

from smartleitstand.cbs_ext.plan import plan

grid = np.zeros([20, 20, 101])
grid[4, 0:16, :] = -1
grid[4, 17:20, :] = -1

# input
agent_pos = [(6, 3), (15, 3), (16, 2), (17, 3)]
jobs = [((1, 6), (9, 6), 0), ((7, 3), (3, 3), 0), ((17, 14), (17, 5), 0)]
idle_goals = [((9, 7), (5, .5)), ((9, 8), (20, .5))]
alloc_jobs = [(0, 1)]

start_time = datetime.datetime.now()

(res_agent_job,
 res_agent_idle,
 res_paths) = plan(agent_pos,
                   jobs,
                   alloc_jobs,
                   idle_goals,
                   grid,
                   plot=False,
                   filename='')

print("computation time:",
      (datetime.datetime.now() - start_time).total_seconds(),
      "s")

print(res_agent_job, res_agent_idle, res_paths)
