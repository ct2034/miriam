import numpy as np
import datetime
import logging
import matplotlib.pyplot as plt
from itertools import *
import pickle

import planner.plan

from planner.plan import plan
from planner.planner_test import get_data_random

agent_n_s = [4, 6, 8, 10]
map_res_s = [10, 11, 12, 13]

results_mean = np.zeros([len(agent_n_s), len(map_res_s)])
results_std = np.zeros([len(agent_n_s), len(map_res_s)])

for agent_n, map_res in product(agent_n_s, map_res_s):
    print("\nCombination agent_n:", agent_n, "map_res", map_res)
    duration = []
    for i in range(5):
        print("test", i)
        agent_pos, grid, idle_goals, jobs = get_data_random(map_res=map_res,
                                                            map_fill_perc=5,
                                                            agent_n=agent_n,
                                                            job_n=agent_n,
                                                            idle_goals_n=agent_n)

        try:
            fname = ''
            fname = planner.plan.pre_calc_paths(jobs, idle_goals, grid)

            start_time = datetime.datetime.now()
            res_agent_job, res_agent_idle, res_paths = planner.plan.plan(agent_pos, jobs, [], idle_goals, grid, filename=fname)

            d = (datetime.datetime.now() - start_time).total_seconds()
            # print("computation time:", d, "s")
            duration.append(d)
            # print("RESULTS:\nres_agent_job", res_agent_job)
            # print("res_agent_idle", res_agent_idle)
            if res_paths is False:
                logging.warning("NO SOLUTION")
                # planner.plan.plot_inputs(agent_pos, idle_goals, jobs, grid, show=True)
            else:
                pass
                # print("res_paths", res_paths)

        except RuntimeError:
            logging.warning("NO SOLUTION (exception)")
            # planner.plan.plot_inputs(agent_pos, idle_goals, jobs, grid, show=True)

    results_mean[agent_n_s.index(agent_n), map_res_s.index(map_res)] = np.mean(duration)
    results_std[agent_n_s.index(agent_n), map_res_s.index(map_res)] = np.std(duration)

with open('/tmp/res.pkl', 'wb') as f:
    pickle.dump((results_mean, results_std), f, pickle.HIGHEST_PROTOCOL)

print("\nmean ..")
print(results_mean)
print("\nstd ..")
print(results_std)