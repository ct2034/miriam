import getpass
import random

import numpy as np
import datetime
from itertools import *

from planner.plan import plan, plot_inputs
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
agents = [(1, 1),
          (3, 3),
          (5, 0),
          (7, 2),
          (9, 1),
          (5, 5),
          (0, 7),
          (9, 7),
          (3, 9),
          (7, 8)]

n_j_s = [2, 4, 6]
n_a_s = [4, 5, 6]

results_mean = np.zeros([len(n_j_s), len(n_a_s)])
results_std = np.zeros([len(n_j_s), len(n_a_s)])

if getpass.getuser() == 'travis':
    n = 200
else:
    n = 1

for n_j, n_a in product(n_j_s, n_a_s):
    print(n_j, n_a)
    duration = []
    for i in range(n):
        print(".")

        jobs = []
        while len(jobs) < n_j:
            j = (random.choice(landmarks), random.choice(landmarks), random.randint(0, 5))
            if j[0] not in map(lambda x: x[0], jobs) and j[1] not in map(lambda x: x[1], jobs) and j[0] != j[1]:
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
        try:
            res_agent_job, res_agent_idle, res_paths = plan(agent_pos, jobs, [], idle_goals, grid, plot=False,
                                                            filename='map_test.pkl')
        except RuntimeError:
            # logging.warning("NO SOLUTION (exception)")
            plot_inputs(agent_pos, idle_goals, jobs, grid, show=True, subplot=111)

        duration.append((datetime.datetime.now() - start_time).total_seconds())

    results_mean[n_j_s.index(n_j), n_a_s.index(n_a)] = np.mean(duration)
    results_std[n_j_s.index(n_j), n_a_s.index(n_a)] = np.std(duration)

print("\nmean ..")
print(results_mean)
print("\nstd ..")
print(results_std)
