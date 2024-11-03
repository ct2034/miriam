import numpy as np

from planner.mapf_implementations.plan_cbs_ta import plan as cbs_ta_plan
from planner.tcbs.plan import plan as tcbs_plan


def tcbs_with_single_goals(gridmap, starts, goals, timeout):
    """
    Adapter to call TCBS with single goals.
    (Normally, TCBS defines jobs as transport tasks,
    i.e., with start and goal locations.)
    """
    # jobs must be transport tasks
    jobs = []
    for g in goals:
        jobs.append((tuple(g), tuple(g), 0))  # start = goal = g, time = 0
    # map must be time-expanded
    t_max = gridmap.shape[0] * gridmap.shape[1] * 2
    _map = np.repeat(gridmap[:, :, np.newaxis], t_max, axis=2)
    return tcbs_plan(
        agent_pos=[tuple(s) for s in starts],
        jobs=jobs,
        alloc_jobs=[],
        idle_goals=[],
        grid=_map,
    )


def execute_both_planners(gridmap, starts, goals, timeout):
    cbs_ta_res = cbs_ta_plan(gridmap, starts, goals, timeout)
    tcbs_res = tcbs_with_single_goals(gridmap, starts, goals, timeout)
    return cbs_ta_res, tcbs_res


def get_cost_from_tcbs_res(tcbs_res):
    """
    Return the sum of costs from TCBS result.

    This must be comparable to the cost from CBS-TA.
    So we have to ignore the last step, where the agents stay at the goal.
    """
    _, _, paths = tcbs_res
    cost = 0
    for path in paths:
        to_start, _ = path
        cost += len(to_start) - 1

    return cost


def get_cost_from_cbs_ta_res(cbs_ta_res):
    """
    Return the sum of costs from CBS-TA result.
    """
    return cbs_ta_res["statistics"]["cost"]


if __name__ == "__main__":
    gridmap = np.array([[0] * 3] * 3)
    starts = [[0, 0], [0, 1], [0, 2]]
    goals = [[2, 0], [2, 1], [1, 2]]
    res = execute_both_planners(gridmap, starts, goals, 10)
    res_cbs_ta, res_tcbs = res

    print("=" * 10)

    print("CBS-TA")
    print(res_cbs_ta)
    print("-" * 10)

    print("Cost from CBS-TA")
    print(get_cost_from_cbs_ta_res(res_cbs_ta))
    print("-" * 10)

    print("TCBS")
    print(res_tcbs)
    print("-" * 10)

    print("Cost from TCBS")
    print(get_cost_from_tcbs_res(res_tcbs))
    print("=" * 10)
