import os
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


def tcbs_with_single_goals(grid, starts, goals):
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
    grid_swapped = grid.swapaxes(0, 1)
    _map = np.repeat(grid_swapped[:, :, np.newaxis], t_max, axis=2)
    # making sure we start with fresh path_save
    fname = "path_save_eval.pkl"
    if os.path.exists(fname):
        os.remove(fname)
    # configuration for TCBS
    config = generate_config()
    config["filename_pathsave"] = fname
    config["time_extended"] = False
    config["save_stats"] = True
    return tcbs_plan(
        agent_pos=[tuple(s) for s in starts],
        jobs=jobs,
        allocated_jobs=[],
        idle_goals=[],
        grid=_map,
        config=config,
    )


def execute_both_planners(grid, starts, goals, timeout):
    """Execute both planners and return the results."""
    if isinstance(goals, np.ndarray):
        goals = [list(g) for g in goals]
    cbs_ta_res = cbs_ta_plan(grid, starts, goals, timeout)
    tcbs_res = tcbs_with_single_goals(grid, starts, goals)
    return cbs_ta_res, tcbs_res


def get_cost_from_tcbs_res(tcbs_res):
    """
    Return the sum of costs from TCBS result.

    This must be comparable to the cost from CBS-TA.
    So we have to ignore the last step, where the agents stay at the goal.
    """
    _, _, paths, _ = tcbs_res
    cost = 0
    for path in paths:
        assert len(path) == 2, (
            "This must not have time-expanded assignments. "
            "(i.e., only one task per agent)"
        )
        to_start, _ = path
        cost += len(to_start) - 1

    return cost


def get_high_level_expanded_from_tcbs_res(tcbs_res):
    """
    Return the high-level expanded nodes from TCBS result.
    """
    _, _, _, stats = tcbs_res
    return stats["n_high_level_expanded"]


def get_low_level_expanded_from_tcbs_res(tcbs_res):
    """
    Return the low-level expanded nodes from TCBS result.
    """
    _, _, _, stats = tcbs_res
    return stats["n_low_level_expanded"]


def get_paths_from_tcbs_res(tcbs_res):
    """
    Return the paths from TCBS result in the format of TCBS result.
    """
    paths = tcbs_res[2]
    paths_out = []
    for ps in paths:
        assert len(ps) == 2, (
            "This must not have time-expanded assignments. "
            "(i.e., only one task per agent)"
        )
        assert len(ps[1]) == 1, "Second element must be a single step"
        assert ps[0][-1][0] == ps[1][0][0], "Last step must be the same (x)"
        assert ps[0][-1][1] == ps[1][0][1], "Last step must be the same (y)"
        paths_out.append((ps[0],))
    return paths_out


def get_cost_from_cbs_ta_res(cbs_ta_res):
    """
    Return the sum of costs from CBS-TA result.
    """
    return cbs_ta_res["statistics"]["cost"]


def get_high_level_expanded_from_cbs_ta_res(cbs_ta_res):
    """
    Return the high-level expanded nodes from CBS-TA result.
    """
    return cbs_ta_res["statistics"]["highLevelExpanded"]


def get_low_level_expanded_from_cbs_ta_res(cbs_ta_res):
    """
    Return the low-level expanded nodes from CBS-TA result.
    """
    return cbs_ta_res["statistics"]["lowLevelExpanded"]


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
    grid = np.array([[FREE] * 5] * 3)
    grid[0, 1] = OBSTACLE
    grid[2, 3] = OBSTACLE
    starts = [[0, 0], [1, 0], [2, 0]]
    goals = [[0, 4], [1, 4], [2, 4]]
    res = execute_both_planners(grid, starts, goals, 10)
    res_cbs_ta, res_tcbs = res

    print("=" * 10)

    print("CBS-TA")
    pprint(res_cbs_ta)

    print("Cost", get_cost_from_cbs_ta_res(res_cbs_ta))
    print("HL expanded", get_high_level_expanded_from_cbs_ta_res(res_cbs_ta))
    print("LL expanded", get_low_level_expanded_from_cbs_ta_res(res_cbs_ta))

    print("-" * 10)

    print("TCBS")
    pprint(res_tcbs)

    print("Cost", get_cost_from_tcbs_res(res_tcbs))
    print("HL expanded", get_high_level_expanded_from_tcbs_res(res_tcbs))
    print("LL expanded", get_low_level_expanded_from_tcbs_res(res_tcbs))

    print("=" * 10)

    plot_comparison_results(grid, res_cbs_ta, res_tcbs)


def plot_comparison_results(grid, res_cbs_ta, res_tcbs):
    fig = plt.figure()
    ax_tcbs = fig.add_subplot(121, projection="3d")
    paths_tcbs = get_paths_from_tcbs_res(res_tcbs)
    plot_results(
        ax_tcbs,
        [],
        paths_tcbs,
        [],
        [],
        grid,
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
        grid,
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
    n_agents = 4
    size = 8

    res_same_cost = 0
    res_success_cbs_ta = 0
    res_success_tcbs = 0

    res_tcbs_cost = []
    res_tcbs_lle = []
    res_tcbs_hle = []
    res_cbs_ta_cost = []
    res_cbs_ta_lle = []
    res_cbs_ta_hle = []

    for _ in tqdm.tqdm(range(n_runs)):
        grid, starts, goals = tracing_paths_in_the_dark(
            size=size,
            fill=0.5,
            n_agents=n_agents,
            rng=rng,
        )
        res = execute_both_planners(grid, starts, goals, 10)
        res_cbs_ta, res_tcbs = res

        if res_cbs_ta != INVALID:
            cost_cbs_ta = get_cost_from_cbs_ta_res(res_cbs_ta)
            res_cbs_ta_cost.append(cost_cbs_ta)
            res_cbs_ta_lle.append(get_low_level_expanded_from_cbs_ta_res(res_cbs_ta))
            res_cbs_ta_hle.append(get_high_level_expanded_from_cbs_ta_res(res_cbs_ta))
            res_success_cbs_ta += 1
        else:
            cost_cbs_ta = INVALID

        if res_tcbs != INVALID:
            cost_tcbs = get_cost_from_tcbs_res(res_tcbs)
            res_tcbs_cost.append(cost_tcbs)
            res_tcbs_lle.append(get_low_level_expanded_from_tcbs_res(res_tcbs))
            res_tcbs_hle.append(get_high_level_expanded_from_tcbs_res(res_tcbs))
            res_success_tcbs += 1
        else:
            cost_tcbs = INVALID

        if cost_cbs_ta == cost_tcbs:
            res_same_cost += 1
        else:
            plot_comparison_results(grid, res_cbs_ta, res_tcbs)

    percentage_same_cost = res_same_cost / n_runs * 100
    print(f"Same cost: {res_same_cost}/{n_runs} = {percentage_same_cost:.2f}%")
    print("CBS-TA")
    print(f"  Success rate: {res_success_cbs_ta/n_runs*100:.2f}%")
    print(f"  Average cost: {np.mean(res_cbs_ta_cost)}")
    print(f"  Average LL expanded: {np.mean(res_cbs_ta_lle)}")
    print(f"  Average HL expanded: {np.mean(res_cbs_ta_hle)}")
    print("TCBS")
    print(f"  Success rate: {res_success_tcbs/n_runs*100:.2f}%")
    print(f"  Average cost: {np.mean(res_tcbs_cost)}")
    print(f"  Average LL expanded: {np.mean(res_tcbs_lle)}")
    print(f"  Average HL expanded: {np.mean(res_tcbs_hle)}")


if __name__ == "__main__":
    demo()
    eval_same_cost_for_random_scenarios()
