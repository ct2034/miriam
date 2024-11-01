import numpy as np

from planner.tcbs.plan import get_paths_for_agent

if __name__ == "__main__":
    vals = {
        "_agent_idle": [(), (), (), ()],
        "_map": np.zeros([10, 10, 100]),
        "agent_job": ((), (), (0, 1), ()),
        "agent_pos": [(4, 2), (5, 4), (3, 5), (5, 6)],
        "alloc_jobs": [],
        "blocks": {},
        "i_a": 2,
        "idle_goals": [
            ((0, 0), (15, 3)),
            ((4, 0), (15, 3)),
            ((9, 0), (15, 3)),
            ((9, 4), (15, 3)),
            ((4, 9), (15, 3)),
            ((0, 9), (15, 3)),
            ((0, 5), (15, 3)),
        ],
        "jobs": [((0, 0), (9, 9), 0.00072), ((0, 0), (9, 9), 0.000309)],
    }

    res = get_paths_for_agent(vals)

    print(res)
