from pprint import pprint
from random import Random

import matplotlib.pyplot as plt
import numpy as np
import tqdm

from definitions import FREE, INVALID, OBSTACLE
from planner.eval.display import plot_results
from planner.mapf_implementations.plan_cbs_ta import plan as cbs_ta_plan
from planner.tcbs.plan import generate_config
from planner.tcbs.plan import plan as tcbs_plan
from scenarios.generators import tracing_paths_in_the_dark


def tcbs_with_single_goals(grid, starts, goals, timeout):
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
    t_max = grid.shape[0] * grid.shape[1] * 2
    _map = np.repeat(grid[:, :, np.newaxis], t_max, axis=2)
    config = generate_config()
    config["time_extended"] = False
    return tcbs_plan(
        agent_pos=[tuple(s) for s in starts],
        jobs=jobs,
        alloc_jobs=[],
        idle_goals=[],
        grid=_map,
        config=config,
    )


def execute_both_planners(grid, starts, goals, timeout):
    """Execute both planners and return the results."""
    if isinstance(goals, np.ndarray):
        goals = [list(g) for g in goals]
    cbs_ta_res = cbs_ta_plan(grid, starts, goals, timeout)
    tcbs_res = tcbs_with_single_goals(grid, starts, goals, timeout)
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
        assert len(path) == 2, (
            "This must not have time-expanded assignments. "
            "(i.e., only one task per agent)"
        )
        to_start, _ = path
        cost += len(to_start) - 1

    return cost


def get_cost_from_cbs_ta_res(cbs_ta_res):
    """
    Return the sum of costs from CBS-TA result.
    """
    return cbs_ta_res["statistics"]["cost"]


def get_paths_from_cbs_ta_res(cbs_ta_res):
    """
    Return the paths from CBS-TA result in the format of TCBS result.
    """
    paths = []
    agents = list(cbs_ta_res["schedule"].keys())
    agents.sort()
    for agent in agents:
        path = []
        for step in cbs_ta_res["schedule"][agent]:
            path.append((step["x"], step["y"], step["t"]))
        paths.append((path,))
    return paths


def demo():
    """Demonstrate the comparison of CBS-TA and TCBS."""
    gridmap = np.array([[FREE] * 3] * 3)
    gridmap[1, 1] = OBSTACLE
    starts = [[0, 0], [0, 1], [0, 2]]
    goals = [[2, 0], [2, 1], [2, 2]]
    res = execute_both_planners(gridmap, starts, goals, 10)
    res_cbs_ta, res_tcbs = res

    print("=" * 10)

    print("CBS-TA")
    pprint(res_cbs_ta)
    print("-" * 10)

    print("Cost from CBS-TA")
    print(get_cost_from_cbs_ta_res(res_cbs_ta))
    print("-" * 10)

    print("TCBS")
    pprint(res_tcbs)
    print("-" * 10)

    print("Cost from TCBS")
    print(get_cost_from_tcbs_res(res_tcbs))
    print("=" * 10)

    fig = plt.figure()
    ax_tcbs = fig.add_subplot(121, projection="3d")

    plot_results(
        ax_tcbs,
        [],
        res_tcbs[2],
        [],
        [],
        gridmap,
        [],
        [],
        "TCBS",
    )

    ax_cbs_ta = fig.add_subplot(122, projection="3d")
    path_cbs_ta = get_paths_from_cbs_ta_res(res_cbs_ta)
    plot_results(
        ax_cbs_ta,
        [],
        path_cbs_ta,
        [],
        [],
        gridmap,
        [],
        [],
        "CBS-TA",
    )

    plt.show()


def eval_same_cost_for_random_scenarios():
    """
    Compare the cost from CBS-TA and TCBS for random scenarios.
    """
    rng = Random(42)
    n_runs = 10
    n_agents = 3
    size = 5

    res_same_cost = 0
    res_success_cbs_ta = 0
    res_success_tcbs = 0

    for _ in tqdm.tqdm(range(n_runs)):
        gridmap, starts, goals = tracing_paths_in_the_dark(
            size=size,
            fill=0.2,
            n_agents=n_agents,
            rng=rng,
        )
        res = execute_both_planners(gridmap, starts, goals, 10)
        res_cbs_ta, res_tcbs = res

        if res_cbs_ta != INVALID:
            cost_cbs_ta = get_cost_from_cbs_ta_res(res_cbs_ta)
            res_success_cbs_ta += 1
        else:
            cost_cbs_ta = INVALID

        if res_tcbs != INVALID:
            cost_tcbs = get_cost_from_tcbs_res(res_tcbs)
            res_success_tcbs += 1
        else:
            cost_tcbs = INVALID

        if cost_cbs_ta == cost_tcbs:
            res_same_cost += 1

    percentage_same_cost = res_same_cost / n_runs * 100
    print(f"Same cost: {res_same_cost}/{n_runs} = {percentage_same_cost:.2f}%")
    print(f"Success CBS-TA: {res_success_cbs_ta/n_runs*100:.2f}%")
    print(f"Success TCBS: {res_success_tcbs/n_runs*100:.2f}%")


if __name__ == "__main__":
    # demo()
    eval_same_cost_for_random_scenarios()
