import datetime

import numpy as np

from planner.cbs_ext.plan import plan_cbsext

grid = np.zeros([20, 20, 101])
grid[4, 0:16, :] = -1
grid[4, 17:20, :] = -1

# input
agent_pos = [(8, 9), (9, 9)]
jobs = [((9, 9), (4, 0), -16),
        ((9, 9), (4, 0), -12)]
idle_goals = [((9, 7), (5, .5)), ((9, 8), (20, .5))]
alloc_jobs = []

start_time = datetime.datetime.now()

(res_agent_job,
 res_agent_idle,
 res_paths) = plan_cbsext(agent_pos,
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
