import numpy as np
import matplotlib.pyplot as plt

from planner.cbs_ext.plan import plan, plot_results
from planner.milp.milp import plan_milp
from tools import load_map


def eval(_map, agents, jobs, fname):
    grid = np.repeat(_map[:, ::2, np.newaxis], 100, axis=2)
    print("CBS EXT")
    res_agent_job, res_agent_idle, res_paths = plan(agents, jobs, [], [], grid, plot=True, filename=fname)
    print("agent_job: " + str(res_agent_job))
    print("paths: " + str(res_paths))

    print("MILP")
    res_agent_job, res_paths = plan_milp(agents, jobs, grid, filename=fname)
    print("agent_job: " + str(res_agent_job))
    print("paths: " + str(res_paths))
    plot_results([], res_paths, agents, res_agent_job, plt.figure(), grid, [], jobs)


_map = load_map('corridor.png')
agents = [(0, 0),
          (0, 1)]
jobs = [((5, 0), (5, 2), 0),
        ((4, 2), (4, 0), 0),
        ((3, 0), (3, 2), 0),
        ]
eval(_map, agents, jobs, 'corridor.pkl')

_map = load_map('mr_t.png')
agents = [(5, 1),
          (5, 3)]
jobs = [((4, 1), (4, 3), 0),
        ((3, 3), (3, 1), 0),
        ]
eval(_map, agents, jobs, 'mr_t.pkl')


_map = load_map('line.png')
agents = [(0, 0),
          (6, 0)]
jobs = [((1, 0), (6, 0), 0),
        ((5, 0), (1, 0), 0),
        ]
eval(_map, agents, jobs, 'line.pkl')
