import numpy as np

from tools import load_map

from planner.strictly_consec.strictly_consec import plan_sc

_map = load_map('../map2.png')
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

plan_sc(agent_pos, jobs, grid)
