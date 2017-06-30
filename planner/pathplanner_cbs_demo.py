import datetime

import numpy as np

from planner.cbs_ext.plan import plan_cbsext

grid = np.zeros([10, 10, 51])
grid[4, 1:9, :] = -1


def pathplan():
    start_time = datetime.datetime.now()
    # This misuses the cbsext planner as cbs only planner by fixing the assignment
    res_agent_job, res_agent_idle, res_paths = plan_cbsext(agent_pos, jobs, [], [], grid,
                                                           plot=False, filename='pathplanning_only.pkl',
                                                           pathplanning_only=True)
    print("computation time:", (datetime.datetime.now() - start_time).total_seconds(), "s")
    for p in res_paths:
        print(p)


# input 1
agent_pos = [(1, 1), (2, 1), (3, 1)]  # three agents
jobs = [((1, 7), (9, 1)), ((1, 8), (8, 1)), ((9, 8),)]  # two jobs 1,6 -> 9,1, 3,3 -> 7,3, a simple goal: 1,1

pathplan()

# input 2
agent_pos = [(1, 1), (2, 1), (3, 1),
             (6, 2), (7, 2), (7, 1)]

jobs = [((1, 7), (9, 1)), ((1, 8), (8, 1)), ((9, 8),),
        ((1, 1), (8, 3)), ((2, 1), (8, 7)), ((1, 2), (9, 9))]

pathplan()
