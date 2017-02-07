import numpy as np
import datetime
import logging

import planner.plan

from planner.plan import plan
from planner.planner_test import get_data_random

for i in range(10):
    print("\nTEST", i)
    agent_pos, grid, idle_goals, jobs = get_data_random(map_res=10,
                                                        map_fill_perc=3,
                                                        agent_n=5,
                                                        job_n=2,
                                                        idle_goals_n=5)

    start_time = datetime.datetime.now()

    try:
        res_agent_job, res_agent_idle, res_paths = planner.plan.plan(agent_pos, jobs, [], idle_goals, grid, filename='')

        print("computation time:", (datetime.datetime.now() - start_time).total_seconds(), "s")
        print("RESULTS:\nres_agent_job", res_agent_job)
        print("res_agent_idle", res_agent_idle)
        if res_paths is False:
            logging.warning("NO SOLUTION")
            planner.plan.plot_inputs(agent_pos, idle_goals, jobs, grid, show=True)
        else:
            print("res_paths", res_paths)

    except RuntimeError:
        logging.warning("NO SOLUTION (exception)")
        planner.plan.plot_inputs(agent_pos, idle_goals, jobs, grid, show=True)
