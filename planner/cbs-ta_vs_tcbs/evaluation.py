import os
from itertools import product
from pprint import pprint
from random import Random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
    res = tcbs_plan(
        agent_pos=[tuple(s) for s in starts],
        jobs=jobs,
        allocated_jobs=[],
        idle_goals=[],
        grid=_map,
        config=config,
    )
    if os.path.exists(fname):
        os.remove(fname)
    return res


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


def barplot_comparison(title_str, n_runs, res_tcbs_cost, res_cbs_ta_cost):
    BAR_WIDTH = 0.4
    plt.figure()
    plt.title(title_str)
    plt.bar(
        range(n_runs),
        res_cbs_ta_cost,
        color="blue",
        label="CBS-TA",
        width=BAR_WIDTH,
    )
    plt.bar(
        np.array(range(n_runs)) + BAR_WIDTH,
        res_tcbs_cost,
        color="red",
        label="TCBS",
        width=BAR_WIDTH,
    )
    plt.legend()


def eval_random_scenarios():
    """
    Compare the cost from CBS-TA and TCBS for random scenarios.
    """
    n_seeds = 10
    n_agents_s = [2, 3, 4]
    fills = [0.4, 0.6, 0.8]
    size = 8

    data = pd.DataFrame(
        columns=["seed", "n_agents", "fill", "cost", "lle", "hle", "success", "planner"]
    )

    n_runs = n_seeds * len(n_agents_s) * len(fills)

    for seed, n_agents, fill in tqdm.tqdm(
        product(range(n_seeds), n_agents_s, fills), total=n_runs
    ):
        rng = Random(seed)
        grid, starts, goals = tracing_paths_in_the_dark(
            size=size,
            fill=fill,
            n_agents=n_agents,
            rng=rng,
        )
        res = execute_both_planners(grid, starts, goals, 10)
        res_cbs_ta, res_tcbs = res

        if res_cbs_ta != INVALID:
            data = pd.concat(
                [
                    data,
                    pd.DataFrame(
                        {
                            "seed": seed,
                            "n_agents": n_agents,
                            "fill": fill,
                            "cost": get_cost_from_cbs_ta_res(res_cbs_ta),
                            "lle": get_low_level_expanded_from_cbs_ta_res(res_cbs_ta),
                            "hle": get_high_level_expanded_from_cbs_ta_res(res_cbs_ta),
                            "success": 1,
                            "planner": "CBS-TA",
                        },
                        index=[0],
                    ),
                ],
                ignore_index=True,
            )
        else:
            data = pd.concat(
                [
                    data,
                    pd.DataFrame(
                        {
                            "seed": seed,
                            "n_agents": n_agents,
                            "fill": fill,
                            "cost": np.nan,
                            "lle": np.nan,
                            "hle": np.nan,
                            "success": 0,
                            "planner": "CBS-TA",
                        },
                        index=[0],
                    ),
                ],
                ignore_index=True,
            )

        if res_tcbs != INVALID:
            data = pd.concat(
                [
                    data,
                    pd.DataFrame(
                        {
                            "seed": seed,
                            "n_agents": n_agents,
                            "fill": fill,
                            "cost": get_cost_from_tcbs_res(res_tcbs),
                            "lle": get_low_level_expanded_from_tcbs_res(res_tcbs),
                            "hle": get_high_level_expanded_from_tcbs_res(res_tcbs),
                            "success": 1,
                            "planner": "TCBS",
                        },
                        index=[0],
                    ),
                ],
                ignore_index=True,
            )
        else:
            data = pd.concat(
                [
                    data,
                    pd.DataFrame(
                        {
                            "seed": seed,
                            "n_agents": n_agents,
                            "fill": fill,
                            "cost": np.nan,
                            "lle": np.nan,
                            "hle": np.nan,
                            "success": 0,
                            "planner": "TCBS",
                        },
                        index=[0],
                    ),
                ],
                ignore_index=True,
            )

        # if not cost_cbs_ta == cost_tcbs:
        #     plot_comparison_results(grid, res_cbs_ta, res_tcbs)

    # write data to file
    data.to_csv("data.csv")


def plot():
    # read data from file
    data = pd.read_csv("data.csv")

    # plot
    sns.set_theme()
    sns.set_context("paper")
    sns.pairplot(
        data,
        hue="planner",
        plot_kws={"alpha": 0.8, "marker": "x"},
    )
    plt.savefig("pairplot.png")
    plt.show()


if __name__ == "__main__":
    eval_random_scenarios()
    plot()
