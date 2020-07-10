import datetime

import numpy as np

from planner.tcbs.plan import plan

if __name__ == "__main__":
    grid = np.zeros([10, 10, 51])
    grid[4, 2:8, :] = -1

    # input
    agent_pos = [(1, 1), (9, 1), (3, 1)]  # three agents
    jobs = [((1, 6), (9, 6), 0), ((3, 3), (7, 3), 0)]  # two jobs 1,6 -> 9,1, 3,3 -> 7,3
    idle_goals = [((9, 7), (5, .5))]  # one idle goal 9,7 with P~N(5,.5)

    start_time = datetime.datetime.now()

    res_agent_job, res_agent_idle, res_paths = plan(agent_pos, jobs, [], idle_goals, grid, plot=True)

    print("computation time:", (datetime.datetime.now() - start_time).total_seconds(), "s")
    print(res_agent_job, res_agent_idle, res_paths)

    assert res_agent_job == ((0,), (), (1,)), "Wrong agent job assignment"
