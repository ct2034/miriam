import datetime
import random

import numpy as np

from planner.tcbs.plan import plan_cbsext
from tools import load_map

if __name__ == "__main__":
    _map = load_map("map.png")
    grid = np.repeat(_map[:, :, np.newaxis], 100, axis=2)

    landmarks = [
        (1, 1),
        (3, 1),
        (5, 1),
        (7, 1),
        (9, 1),
        (8, 5),
        (1, 6),
        (9, 7),
        (1, 8),
        (9, 9),
    ]

    # input
    agents = [
        (1, 1),
        (3, 3),
        (5, 0),
        (7, 2),
        (9, 1),
        (5, 5),
        (0, 7),
        (9, 7),
        (3, 9),
        (7, 8),
    ]

    n_j = 4
    n_a = 4

    jobs = []
    while len(jobs) < n_j:
        j = (random.choice(landmarks), random.choice(landmarks), random.randint(0, 5))
        jobs.append(j)

    idle_goals = []
    for i_l in range(len(landmarks)):
        idle_goals.append(
            (landmarks[i_l], (random.randint(0, 20), random.randint(1, 50) / 10))
        )
    agent_pos = []
    while len(agent_pos) < n_a:
        a = random.choice(agents)
        if a not in agent_pos:
            agent_pos.append(a)

    start_time = datetime.datetime.now()
    res_agent_job, res_agent_idle, res_paths = plan_cbsext(
        agent_pos, jobs, [], idle_goals, grid, plot=True, filename="../map_test.pkl"
    )
