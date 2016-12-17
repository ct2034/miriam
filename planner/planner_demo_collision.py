import numpy as np
import datetime

from planner.plan import plan

grid = np.zeros([20, 20, 101])
grid[4, 0:16, :] = -1
grid[4, 17:20, :] = -1

# input
agent_pos = [(1, 1), (15, 2), (16, 1), (17, 2)]
jobs = [((1, 6), (9, 6)), ((7, 3), (3, 3))]
idle_goals = [((9, 7), (5, .5)), ((9, 8), (20, .5))]

start_time = datetime.datetime.now()

(res_agent_job,
 res_agent_idle,
 res_paths) = plan(agent_pos,
                   jobs,
                   idle_goals,
                   grid,
                   plot=True,
                   filename='')

print("computation time:",
      (datetime.datetime.now() - start_time).total_seconds(),
      "s")

print(res_agent_job, res_agent_idle, res_paths)
