import datetime

import numpy as np

from planner.cbs_ext.plan import plan

grid = np.zeros([10, 10, 51])
grid[4, 2:8, :] = -1

# input
agent_pos = [(1, 1), (9, 1), (3, 1)]  # three agents
jobs = [((1, 6), (9, 6), 0), ((3, 3), (7, 3), 0)]  # two jobs 1,6 -> 9,1, 3,3 -> 7,3
idle_goals = [((9, 7), (5, .5))]  # one idle goal 9,7 with P~N(5,.5)

# expected results
agent_job = [(0, 0), (2, 1)]
agent_idle = [(1, 0)]

start_time = datetime.datetime.now()

res_agent_job, res_agent_idle, res_paths = plan(agent_pos, jobs, [], idle_goals, grid, plot=False)

print("computation time:", (datetime.datetime.now() - start_time).total_seconds(), "s")

print(res_agent_job, res_agent_idle, res_paths)
