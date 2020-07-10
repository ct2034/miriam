import datetime

import logging
import numpy as np

from planner.tcbs.plan import plan as plan_cbsext
# from planner.milp.milp import plan_milp
from tools import load_map

if __name__ == "__main__":
        logging.getLogger('pyomo').setLevel(logging.INFO)
        logging.getLogger('pyutilib.component.core.pca').setLevel(logging.INFO)

        _map = load_map('map2.png')
        _map = _map[:, ::2]
        grid = np.repeat(_map[:, :, np.newaxis], 100, axis=2)

        # input
        agent_pos = [(0, 0), (3, 0), (2, 1)]
        jobs = [((0, 8), (0, 2), 0),
                ((1, 8), (2, 4), 0),
                # ((2, 8), (4, 8), 0),
                ((7, 6), (3, 8), 0),
                ((8, 7), (8, 2), 0)]
        idle_goals = []

        # print("MILP")
        # start_time = datetime.datetime.now()
        # res_agent_job, res_paths = plan_milp(agent_pos, jobs, grid, filename='map2_test.pkl')
        # print("computation time:", (datetime.datetime.now() - start_time).total_seconds(), "s")
        # print(res_agent_job, res_paths)

        print("CBSEXT")
        start_time = datetime.datetime.now()
        res_agent_job, res_agent_idle, res_paths = plan_cbsext(agent_pos, jobs, [], idle_goals, grid, plot=True,
                                                        filename='map2_test.pkl')
        print("computation time:", (datetime.datetime.now() - start_time).total_seconds(), "s")
        print(res_agent_job, res_agent_idle, res_paths)
