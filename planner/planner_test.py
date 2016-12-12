import numpy as np
import planner.plan


def test_basic():
    grid = np.zeros([10, 10])
    grid[4, 2:8] = -1

    # input
    agent_pos = [(1, 1), (9, 1), (3, 1)]  # three agents
    jobs = [((1, 6), (9, 6)), ((3, 3), (7, 3))]  # two jobs 1,6 -> 9,1, 3,3 -> 7,3
    idle_goals = [((9, 7), (5, .5))]  # one idle goal 9,7 with P~N(5,.5)

    # expected results
    agent_job = [(0, 0)]
    agent_idle = [(1, 0)]

    res_agent_job, res_agent_idle = planner.plan.plan(agent_pos, jobs, idle_goals, grid)

    assert res_agent_job == agent_job, "wrong agent -> job assignment"
    assert res_agent_idle == agent_idle, "wrong agent -> idle_goal assignment"
