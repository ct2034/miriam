import datetime

import numpy as np

from planner.cbs_ext.plan import plan


def values_vertexswap():
    grid = np.zeros([7, 7, 101])
    # input
    agent_pos = [(0, 3),
                 (5, 3)]
    jobs = [((0, 3), (5, 3), 0),
            ((5, 3), (0, 3), 0)]
    idle_goals = []
    alloc_jobs = [(0, 0), (1, 1)]
    start_time = datetime.datetime.now()
    return grid, agent_pos, jobs, idle_goals, alloc_jobs, start_time


if __name__ == "__main__":
    grid, agent_pos, jobs, idle_goals, alloc_jobs, start_time = values_vertexswap()

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
