import datetime
import getpass
import random
from itertools import *

import numpy as np

from planner.cbs_ext.plan import plan, plot_inputs
from tools import load_map

_map = load_map('corridor.png')
grid = np.repeat(_map[:, ::2, np.newaxis], 100, axis=2)

# input
agents = [(0, 0),
          (0, 1)]
jobs = [((8, 0), (8, 2), 0),
        ((7, 2), (7, 0), 0),
        ((6, 0), (6, 2), 0),
        ]

res_agent_job, res_agent_idle, res_paths = plan(agents, jobs, [], [], grid, plot=False)

print("agent_job: " + str(res_agent_job))
print("paths: " + str(res_paths))
