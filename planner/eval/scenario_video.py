import numpy as np
import matplotlib.pyplot as plt

from planner.eval.display import animate_results
from tools import load_map

(tcbs_paths, tcbs_agent_job, minlp_paths, minlp_agent_job, agent_pos, jobs) = (
    [
        (
            [(6, 3, 0), (5, 3, 1)],
            [(5, 3, 2), (6, 3, 3), (6, 2, 4), (6, 1, 5), (5, 1, 6)],
            [(5, 1, 7), (4, 1, 8)],
            [(4, 1, 9), (5, 1, 10), (6, 1, 11), (6, 2, 12), (6, 3, 13), (5, 3, 14), (4, 3, 15)]
        ),
        (
            [(3, 1, 0), (3, 1, 1), (3, 1, 2), (3, 1, 3), (3, 1, 4), (3, 1, 5), (3, 1, 6), (3, 1, 7), (3, 1, 8),
             (3, 1, 9),
             (3, 1, 10), (3, 1, 11), (3, 1, 12), (3, 1, 13), (3, 1, 14), (3, 1, 15)],
        )
    ],
    ((0, 1), ()),
    [
        (
            [(6, 3, 0), (5, 3, 1)],
            [(5, 3, 2), (6, 3, 3), (6, 2, 4), (6, 1, 5), (5, 1, 6)],
            [(5, 1, 7), (5, 1, 8), (5, 1, 9), (5, 1, 10), (5, 1, 11), (5, 1, 12), (5, 1, 13),
             (5, 1, 14), (5, 1, 15), (5, 1, 16), (5, 1, 17), (5, 1, 18), (5, 1, 19), (5, 1, 20)]
        ),
        (
            [(3, 1, 0), (4, 1, 1)],
            [(4, 1, 2), (3, 1, 3), (3, 0, 4), (2, 0, 5), (1, 0, 6), (1, 1, 7), (1, 2, 8), (1, 3, 9), (1, 4, 10),
             (1, 5, 11),
             (1, 6, 12), (1, 7, 13), (2, 7, 14), (3, 7, 15), (3, 6, 16), (3, 5, 17), (3, 4, 18), (3, 3, 19), (4, 3, 20)]
        )
    ],
    [(0,), (1,)],
    [(6, 3), (3, 1)],
    [((5, 3), (5, 1), 0), ((4, 1), (4, 3), 0)]
)

grid = load_map('mr_t.png')
agent_pos = [(6, 3),
             (3, 1)]
jobs = [((5, 3), (5, 1), 0),
        ((4, 1), (4, 3), 0)]

grid = np.repeat(grid[:, ::2, np.newaxis], 100, axis=2)

f = plt.figure()
f.set_size_inches(8, 4.5)
ani = animate_results(
    f, [], tcbs_paths, tcbs_agent_job, agent_pos, grid, [], jobs, 'TCBS'
)
ani.save("video/scenario_tcbs.mp4", writer="ffmpeg", fps=10)

f = plt.figure()
f.set_size_inches(8, 4.5)
ani = animate_results(
    f, [], minlp_paths, minlp_agent_job, agent_pos, grid, [], jobs, 'MINLP'
)
ani.save("video/scenario_minlp.mp4", writer="ffmpeg", fps=10)
# plt.show()
