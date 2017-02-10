import random

import numpy as np
import datetime
from itertools import *

from planner.plan import plan
from planner.planner_test import load_map

_map = load_map('planner/map.png')
grid = np.repeat(_map[:, :, np.newaxis], 100, axis=2)

landmarks = [(1, 1),
             (3, 1),
             (5, 1),
             (7, 1),
             (9, 1),
             (8, 5),
             (1, 6),
             (9, 7),
             (1, 8),
             (9, 9)]

# input
agent_pos = [(1, 1),
             (3, 3),
             (5, 0),
             (7, 2),
             (9, 1),
             (5, 5),
             (0, 7),
             (9, 7),
             (3, 9),
             (7, 8)]

n_j = 4
jobs = []
while len(jobs) < n_j:
    j = (random.choice(landmarks), random.choice(landmarks), random.randint(0, 5))
    if j[0] not in map(lambda x: x[0], jobs) and j[1] not in map(lambda x: x[1], jobs):
        jobs.append(j)

idle_goals = []
for i in range(len(landmarks)) - 6:
    idle_goals.append(
        (landmarks[i], (random.randint(0, 20), random.randint(1, 50)/10))
    )

start_time = datetime.datetime.now()

res_agent_job, res_agent_idle, res_paths = plan(agent_pos, jobs, [], idle_goals, grid, plot=False)

print("computation time:", (datetime.datetime.now() - start_time).total_seconds(), "s")

print(res_agent_job, res_agent_idle, res_paths)
