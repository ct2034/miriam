from planner.tcbs.plan import generate_config
from planner.greedy.greedy import plan_greedy
import numpy as np


# def test_greedy_simple():
#     res_agent_job, res_paths = plan_greedy(
#         [(1, 1), (2, 2)],
#         [((3, 3), (1, 4), 0.), ((4, 1), (0, 0), 0.)],
#         np.zeros([5, 5, 100]),
#         generate_config()
#     )
#     assert res_agent_job, "No result"
#     assert res_paths, "No result"
