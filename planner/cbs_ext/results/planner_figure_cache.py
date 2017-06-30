import datetime
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from planner.cbs_ext.plan import plan
from planner.planner_test import get_data_random

plt.style.use('bmh')
fname = "/tmp/test.pkl"

n_tests = 10
n_agent = [1, 2, 3, 4, 5]

res = np.zeros([len(n_agent), n_tests])
res2 = np.zeros([len(n_agent), n_tests])

for i_agents in range(len(n_agent)):
    print("i_agents", i_agents)
    for i_test in range(n_tests):
        print("i_test", i_test)
        if os.path.exists(fname):
            os.remove(fname)
        assert not os.path.exists(fname), "File exists already"

        agent_pos, grid, idle_goals, jobs = get_data_random(10, 5, n_agent[i_agents], n_agent[i_agents], 5)

        start_time = datetime.datetime.now()
        try:
            plan(agent_pos, jobs, [], idle_goals, grid, filename=fname)
        except RuntimeError:
            print("NO SOLUTION")
            pass
        time1 = (datetime.datetime.now() - start_time).total_seconds()
        assert os.path.isfile(fname), "Algorithm has not created a file"

        start_time = datetime.datetime.now()
        try:
            plan(agent_pos, jobs, [], idle_goals, grid, filename=fname)
        except RuntimeError:
            print("NO SOLUTION")
            pass
        time2 = (datetime.datetime.now() - start_time).total_seconds()

        os.remove(fname)
        assert not os.path.exists(fname), "File exists after delete"

        res[i_agents, i_test] = (time1 - time2) / time1 * 100
        res2[i_agents, i_test] = time2

try:
    with open('figure_cache.pkl', 'wb') as f:
        pickle.dump((res, res2), f, pickle.HIGHEST_PROTOCOL)
except Exception as e:
    print(e)

plt.figure()
plt.boxplot(res.T)
plt.xlabel('Agents')
plt.ylabel('Planning Time Saving [%]')

plt.savefig('figure_cache.png', bbox_inches='tight')

plt.figure()
plt.boxplot(res2.T)
plt.xlabel('Agents')
plt.ylabel('Planning Time [s]')

plt.savefig('figure_planningtime.png', bbox_inches='tight')
