#!/usr/bin/env python3
import logging
import pickle
from collections import OrderedDict
from copy import copy
from functools import lru_cache
from itertools import product, repeat
from typing import *

import matplotlib.colors as colors
import numpy as np
import pandas as pd
from definitions import INVALID, NO_SUCCESS, SUCCESS
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tools import BLUE_SEQ, RESET_SEQ, ProgressBar

from scenarios.evaluators import *
from scenarios.generators import *

GENERATORS = 'generators'
FILLS = 'fills'
N_AGENTSS = 'n_agentss'
EVALUATIONS = 'evaluations'
# -------------------------
WELL_FORMED = "well_formed"
DIFF_INDEP = "diff_indep"
DIFF_SIM_DECEN = "diff_sim_decen"
ECBS_SUCCESS = "ecbs_success"
ECBS_COST = "ecbs_cost"
ECBS_VERTEX_BLOCKS = "ecbs_vertex_blocks"
ECBS_EDGE_BLOCKS = "ecbs_edge_blocks"
ECBS_EXPANDED_NODES = "ecbs_expanded_nodes"
USEFULLNESS = "usefullness"
ICTS_SUCCESS = "icts_success"
ICTS_COST = "icts_cost"
ICTS_EXPANDED_NODES = "icts_expanded_nodes"
DIFFERENCE_ECBS_EN_MINUS_ICTS_EN = "difference_ecbs_en_-_icts_en"
N_NODES = "n_nodes"
N_EDGES = "n_edges"
MEAN_DEGREE = "mean_degree"
N_NODES_TA = "n_nodes_ta"
N_EDGES_TA = "n_edges_ta"


def init_values_debug():
    size = 8  # size for all scenarios
    n_fills = 2  # how many different fill values there should be
    n_n_agentss = 2  # how many different numbers of agents should there be"""
    n_runs = 2  # how many runs per configuration
    max_fill = .4  # maximal fill to sample until
    low_agents = 1  # lowest number of agents
    high_agents = 5  # highest number of agents
    return (max_fill, n_fills, n_n_agentss, n_runs, size,
            low_agents, high_agents)


def init_values_main():
    size = 8  # size for all scenarios
    n_fills = 8  # how many different fill values there should be
    n_n_agentss = 8  # how many different numbers of agents should there be"""
    n_runs = 32  # how many runs per configuration
    max_fill = .6  # maximal fill to sample until
    low_agents = 1  # lowest number of agents
    high_agents = 16  # highest number of agents
    return (max_fill, n_fills, n_n_agentss, n_runs, size,
            low_agents, high_agents)


def init_values_focus():
    size = 10  # size for all scenarios
    n_fills = 6  # how many different fill values there should be
    n_n_agentss = 6  # how many different numbers of agents should there be"""
    n_runs = 32  # how many runs per configuration
    max_fill = .7  # maximal fill to sample until
    low_agents = 5  # lowest number of agents
    high_agents = 10  # highest number of agents
    return (max_fill, n_fills, n_n_agentss, n_runs, size,
            low_agents, high_agents)


def get_fname(generator_name, evaluation, extension):
    return ("scenarios/res_" + generator_name + "-" +
            evaluation + "." + extension)


def get_fname_both(evaluation, extension):
    return ("scenarios/res_both-" +
            evaluation + "." + extension)


def genstr(generator):
    generator_name = str(generator).split("at 0x")[
        0].replace("<function ", "").replace(" ", "")
    return generator_name


def add_colums(df1: pd.DataFrame, df2: pd.DataFrame):
    """Assuming df2 has only one column,
    it will be added as new column to df1."""
    assert len(df2.columns) == 1
    assert df2.columns[0] not in df1.columns
    df1[df2.columns[0]] = df2


def plot_images(
        results: List[pd.DataFrame], titles: List[str],
        title: Callable, n_agentss: List[int], fills: List[float],
        evaluation: str, normalize_cbars: str = ""):
    n_fills = len(fills)
    n_n_agentss = len(n_agentss)

    # our cmap with support for over / under
    palette = copy(plt.cm.viridis)
    palette.set_over('w', 1.0)
    palette.set_under('k', 1.0)
    palette.set_bad('k', 1.0)

    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(title, fontsize=16)
    subplot_rows = 2
    subplot_cols = int(np.ceil(len(results) / 2))

    if normalize_cbars == "expanded":
        # same min / max for top bottom plots
        mins = [0]*subplot_cols
        maxs = [0]*subplot_cols
        for i, r in enumerate(results):
            this_data = r.to_numpy()
            this_min = np.min(this_data[np.logical_not(np.isnan(this_data))])
            this_max = np.max(this_data[np.logical_not(np.isnan(this_data))])
            if i < subplot_cols:
                mins[i] = this_min
                maxs[i] = this_max
            else:
                mins[i-subplot_cols] = min(this_min, mins[i-subplot_cols])
                maxs[i-subplot_cols] = max(this_max, maxs[i-subplot_cols])

        # same min / max for ecbs and icts expanded nodes
        for i in [0, 1]:
            mins[i] = min(mins[0], mins[1])
            maxs[i] = max(maxs[0], maxs[1])

    for i, r in enumerate(results):
        # do we have any results?
        assert not np.all(r == 0)
        assert not np.all(r == -1)
        r_final = np.full([n_fills, n_n_agentss], np.nan, dtype=float)
        for i_f, i_a in product(range(n_fills),
                                range(n_n_agentss)):
            fill = fills[i_f]
            n_agents = n_agentss[i_a]
            this_data = r.loc[(fill, n_agents)].to_numpy()
            if len(this_data[np.logical_not(np.isnan(this_data))]) > 0:
                r_final[i_f, i_a] = np.mean(
                    this_data[np.logical_not(np.isnan(this_data))]
                )
        ax = fig.add_subplot(subplot_rows, subplot_cols, i+1)
        im = ax.imshow(
            r_final,
            cmap=palette,
            norm=colors.SymLogNorm(1, vmin=mins[i % subplot_cols],
                                   vmax=maxs[i % subplot_cols]),
            origin='lower'
        )
        fig.colorbar(im, extend='both', spacing='uniform',
                     shrink=0.9, ax=ax)
        plt.title(titles[i])
        plt.ylabel('Fills')
        plt.yticks(range(n_fills), map(lambda a: str(a), fills))
        plt.xlabel('Agents')
        plt.xticks(range(n_n_agentss), map(
            lambda a: str(int(a)), n_agentss))

    plt.tight_layout()
    fname = get_fname(title, evaluation, "png")
    plt.savefig(fname)


def plot_scatter(df, xs, ys, cs, titles, title="scatter"):
    assert len(xs) == len(titles)
    n_plots = len(xs) * len(ys)
    rows = 2
    columns = int(np.ceil(float(n_plots)/rows))

    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(title, fontsize=16)
    i_p = 1
    cm = plt.cm.viridis
    colors = df.index.get_level_values(
        level=cs).to_numpy()
    unique_colors = np.unique(colors)
    df_cols = len(df.columns)
    for i_xs, i_ys in product(range(len(xs)), range(len(ys))):
        x = xs[i_xs]
        y = ys[i_ys]
        dx = df.xs(x, level=EVALUATIONS).to_numpy().flatten()
        dy = df.xs(y, level=EVALUATIONS).to_numpy().flatten()
        dc = df.xs(x, level=EVALUATIONS).index.get_level_values(
            level=cs)
        c = cm(np.array([np.where(unique_colors == ic)
                         for ic in dc]).repeat(df_cols) / len(unique_colors))
        ax = fig.add_subplot(rows, columns, i_p)
        i_p += 1
        ax.scatter(
            dx, dy,
            c=c,
            marker="."
        )
        i_nn = np.isfinite(dx) & np.isfinite(dy)
        coeff = np.polyfit(dx[i_nn], dy[i_nn], 1)
        poly = np.poly1d(coeff)
        plt.plot(dx[i_nn], poly(dx[i_nn]), "r--")
        print("%s y=%.6fx+(%.6f)" % (x, coeff[0], coeff[1]))
        plt.title(titles[i_xs])
        plt.xlabel(x)
        plt.ylabel(y)
    plt.tight_layout()
    fname = get_fname(title, "ecbs_icts", "png")
    plt.savefig(fname)


def main_icts():
    # no warnings pls
    logging.getLogger('sim.decentralized.agent').setLevel(logging.ERROR)

    # (max_fill, n_fills, n_n_agentss, n_runs, size,
    #  low_agents, high_agents) = init_values_debug()
    # (max_fill, n_fills, n_n_agentss, n_runs, size,
    #  low_agents, high_agents) = init_values_main()
    (max_fill, n_fills, n_n_agentss, n_runs, size,
     low_agents, high_agents) = init_values_focus()

    # parameters to evaluate against
    generators = [
        # like_policylearn_gen,
        like_sim_decentralized,
        tracing_pathes_in_the_dark,
        building_walls
    ]

    fills = np.around(
        np.linspace(0, max_fill, n_fills),
        2
    )  # list of fills we want

    n_agentss = np.linspace(low_agents, high_agents, n_n_agentss, dtype=int
                            )  # list of different numbers of agents we want

    evaluations = [
        ECBS_SUCCESS,
        ECBS_COST,
        ECBS_EXPANDED_NODES,
        ECBS_VERTEX_BLOCKS,
        ECBS_EDGE_BLOCKS,
        ICTS_COST,
        ICTS_SUCCESS,
        ICTS_EXPANDED_NODES,
        DIFFERENCE_ECBS_EN_MINUS_ICTS_EN,
        N_NODES,
        N_EDGES,
        MEAN_DEGREE,
        N_NODES_TA,
        N_EDGES_TA
    ]

    # preparing panda dataframes
    index_arrays = {
        GENERATORS: list(map(genstr, generators)),
        FILLS: fills,
        N_AGENTSS: n_agentss,
        EVALUATIONS: evaluations
    }
    idx = pd.MultiIndex.from_product(
        index_arrays.values(), names=index_arrays.keys())
    df_results = pd.DataFrame(index=idx)

    df_results.sort_index(inplace=True)
    assert len(index_arrays) == df_results.index.lexsort_depth

    pdo = ProgressBar(BLUE_SEQ + "Overall" + RESET_SEQ, n_runs)
    for i_r in range(n_runs):
        df_col = evaluate_full(
            i_r, idx, generators, fills, n_agentss, size)
        add_colums(df_results, df_col)
        pdo.progress()
    pdo.end()

    # with pd.option_context('display.max_rows',
    #                        None,
    #                        'display.max_columns',
    #                        None):  # all rows and columns
    #     print(df_results)
    print(df_results.info)

    df_results.to_pickle(get_fname_both("icts", "pkl"))

    # compare expanded nodes
    # data_to_print = OrderedDict()
    # for gen in generators:
    #     genname = genstr(gen)
    #     data_to_print[genname+" ECBS expanded nodes"] = df_results.loc[
    #         (genname)].xs(ECBS_EXPANDED_NODES, level=EVALUATIONS)
    #     data_to_print[genname+" ICTS expanded nodes"] = df_results.loc[
    #         (genname)].xs(ICTS_EXPANDED_NODES, level=EVALUATIONS)
    #     data_to_print[genname+" Difference ECBS minus ICTS expanded nodes"
    #                   ] = df_results.loc[(genname)].xs(
    #                       DIFFERENCE_ECBS_EN_MINUS_ICTS_EN,
    #         level=EVALUATIONS)
    #     # plot
    # plot_images(
    #     list(data_to_print.values()),
    #     list(data_to_print.keys()),
    #     title="expanded-nodes",
    #     n_agentss=n_agentss,
    #     fills=fills,
    #     evaluation="icts",
    #     normalize_cbars="expanded"
    # )

    # compare expanded nodes over graph properties
    xs = [N_NODES, N_EDGES, MEAN_DEGREE, N_NODES_TA, N_EDGES_TA]
    ys = [DIFFERENCE_ECBS_EN_MINUS_ICTS_EN]
    plot_scatter(df_results, xs, ys, cs=GENERATORS, titles=xs)

    plt.show()


def evaluate_full(i_r, idx, generators, fills, n_agentss, size):
    col_name = "seed{}".format(i_r)
    pb = ProgressBar("column >{}<".format(col_name),
                     (len(generators) * len(fills) * len(n_agentss)), 10)
    # Taking care of some little pandas ...........................
    df_col = pd.DataFrame(index=idx, dtype=float)
    df_col[col_name] = [np.nan] * len(df_col.index)
    df_col.sort_index(inplace=True)
    for gen, i_f, i_a in product(generators,
                                 range(len(fills)),
                                 range(len(n_agentss))):
        pb.progress()
        # generating a scenario .......................................
        fill = fills[i_f]
        n_agents = n_agentss[i_a]
        env, starts, goals = gen(
            size, fill, n_agents, seed=i_r)
        # graph based metrics .........................................
        df_col.loc[(genstr(gen), fill, n_agents, N_NODES), col_name
                   ] = n_nodes(env)
        df_col.loc[(genstr(gen), fill, n_agents, N_EDGES), col_name
                   ] = n_edges(env)
        df_col.loc[(genstr(gen), fill, n_agents, MEAN_DEGREE), col_name
                   ] = mean_degree(env)
        df_col.loc[(genstr(gen), fill, n_agents, N_NODES_TA), col_name
                   ] = n_nodes(env) * n_agents
        df_col.loc[(genstr(gen), fill, n_agents, N_EDGES_TA), col_name
                   ] = n_edges(env) * n_agents
        # calculating optimal cost ....................................
        if i_f > 0 and df_col.loc[
                (genstr(gen), fills[i_f-1], n_agents, ECBS_SUCCESS), col_name
        ] == NO_SUCCESS:
            # previous fills timed out
            res_ecbs = INVALID
        elif i_a > 0 and df_col.loc[
                (genstr(gen), fill, n_agentss[i_a-1], ECBS_SUCCESS), col_name
        ] == NO_SUCCESS:
            # previous agent count failed as well
            res_ecbs = INVALID
        else:
            res_ecbs = cost_ecbs(env, starts, goals)
        if res_ecbs == INVALID:
            df_col.loc[
                (genstr(gen), fill, n_agents,
                 ECBS_SUCCESS), col_name] = NO_SUCCESS
        else:  # valid ecbs result
            df_col.loc[
                (genstr(gen), fill, n_agents, ECBS_SUCCESS),
                col_name] = SUCCESS
            df_col.loc[
                (genstr(gen), fill, n_agents, ECBS_COST),
                col_name] = res_ecbs
            df_col.loc[
                (genstr(gen), fill, n_agents, ECBS_EXPANDED_NODES), col_name
            ] = expanded_nodes_ecbs(
                env, starts, goals)
            # evaluating blocks
            blocks = blocks_ecbs(env, starts, goals)
            if blocks != INVALID:
                (
                    df_col.loc[
                        (genstr(gen), fill, n_agents,
                         ECBS_VERTEX_BLOCKS), col_name],
                    df_col.loc[
                        (genstr(gen), fill, n_agents,
                         ECBS_EDGE_BLOCKS), col_name]
                ) = blocks
        # what is icts cost? ..........................................
        if i_f > 0 and df_col.loc[
                (genstr(gen), fills[i_f-1], n_agents, ICTS_SUCCESS), col_name
        ] == NO_SUCCESS:
            # previous fills timed out
            res_icts = INVALID
        elif i_a > 0 and df_col.loc[
                (genstr(gen), fill, n_agentss[i_a-1], ICTS_SUCCESS), col_name
        ] == NO_SUCCESS:
            # previous agent count failed as well
            res_icts = INVALID
        else:
            res_icts = cost_icts(env, starts, goals)
        if res_icts == INVALID:
            df_col.loc[
                (genstr(gen), fill, n_agents, ICTS_SUCCESS),
                col_name] = NO_SUCCESS
        else:
            df_col.loc[
                (genstr(gen), fill, n_agents, ICTS_SUCCESS),
                col_name] = SUCCESS
            df_col.loc[
                (genstr(gen), fill, n_agents, ICTS_COST), col_name] = res_icts
            df_col.loc[
                (genstr(gen), fill, n_agents, ICTS_EXPANDED_NODES), col_name
            ] = expanded_nodes_icts(
                env, starts, goals)
        # run icts and compare n of expanded nodes
        if (
            not np.isnan(df_col.loc[
                (genstr(gen), fill, n_agents, ECBS_EXPANDED_NODES), col_name])
            and
            not np.isnan(df_col.loc[
                (genstr(gen), fill, n_agents, ICTS_EXPANDED_NODES), col_name])
        ):
            df_col.loc[
                (genstr(gen), fill, n_agents,
                 DIFFERENCE_ECBS_EN_MINUS_ICTS_EN),
                col_name
            ] = float(
                df_col.loc[
                    (genstr(gen), fill, n_agents, ECBS_EXPANDED_NODES),
                    col_name
                ] -
                df_col.loc[
                    (genstr(gen), fill, n_agents, ICTS_EXPANDED_NODES),
                    col_name
                ]
            )
    pb.end()
    return df_col


def evaluate_en_comparison(i_r, idx, generators, fills, n_agentss, size):
    col_name = "seed{}".format(i_r)
    pb = ProgressBar("column >{}<".format(col_name),
                     (len(generators) * len(fills) * len(n_agentss)), 10)
    # Taking care of some little pandas ...........................
    df_col = pd.DataFrame(index=idx, dtype=float)
    df_col[col_name] = [np.nan] * len(df_col.index)
    df_col.sort_index(inplace=True)
    for gen, i_f, i_a in product(generators,
                                 range(len(fills)),
                                 range(len(n_agentss))):
        pb.progress()
        # generating a scenario .......................................
        fill = fills[i_f]
        n_agents = n_agentss[i_a]
        env, starts, goals = gen(
            size, fill, n_agents, seed=i_r)
        # calculating optimal cost ....................................
        if i_f > 0 and df_col.loc[
                (genstr(gen), fills[i_f-1], n_agents, ECBS_SUCCESS), col_name
        ] == NO_SUCCESS:
            # previous fills timed out
            res_ecbs = INVALID
        elif i_a > 0 and df_col.loc[
                (genstr(gen), fill, n_agentss[i_a-1], ECBS_SUCCESS), col_name
        ] == NO_SUCCESS:
            # previous agent count failed as well
            res_ecbs = INVALID
        else:
            res_ecbs = cost_ecbs(env, starts, goals)
        if res_ecbs == INVALID:
            df_col.loc[
                (genstr(gen), fill, n_agents,
                 ECBS_SUCCESS), col_name] = NO_SUCCESS
        else:  # valid ecbs result
            df_col.loc[
                (genstr(gen), fill, n_agents, ECBS_SUCCESS),
                col_name] = SUCCESS
            df_col.loc[
                (genstr(gen), fill, n_agents, ECBS_COST),
                col_name] = res_ecbs
            df_col.loc[
                (genstr(gen), fill, n_agents, ECBS_EXPANDED_NODES), col_name
            ] = expanded_nodes_ecbs(
                env, starts, goals)
            # evaluating blocks
            blocks = blocks_ecbs(env, starts, goals)
            if blocks != INVALID:
                (
                    df_col.loc[
                        (genstr(gen), fill, n_agents,
                         ECBS_VERTEX_BLOCKS), col_name],
                    df_col.loc[
                        (genstr(gen), fill, n_agents,
                         ECBS_EDGE_BLOCKS), col_name]
                ) = blocks
        # what is icts cost? ..........................................
        if i_f > 0 and df_col.loc[
                (genstr(gen), fills[i_f-1], n_agents, ICTS_SUCCESS), col_name
        ] == NO_SUCCESS:
            # previous fills timed out
            res_icts = INVALID
        elif i_a > 0 and df_col.loc[
                (genstr(gen), fill, n_agentss[i_a-1], ICTS_SUCCESS), col_name
        ] == NO_SUCCESS:
            # previous agent count failed as well
            res_icts = INVALID
        else:
            res_icts = cost_icts(env, starts, goals)
        if res_icts == INVALID:
            df_col.loc[
                (genstr(gen), fill, n_agents, ICTS_SUCCESS),
                col_name] = NO_SUCCESS
        else:
            df_col.loc[
                (genstr(gen), fill, n_agents, ICTS_SUCCESS),
                col_name] = SUCCESS
            df_col.loc[
                (genstr(gen), fill, n_agents, ICTS_COST), col_name] = res_icts
            df_col.loc[
                (genstr(gen), fill, n_agents, ICTS_EXPANDED_NODES), col_name
            ] = expanded_nodes_icts(
                env, starts, goals)
        # run icts and compare n of expanded nodes
        if (
            not np.isnan(df_col.loc[
                (genstr(gen), fill, n_agents, ECBS_EXPANDED_NODES), col_name])
            and
            not np.isnan(df_col.loc[
                (genstr(gen), fill, n_agents, ICTS_EXPANDED_NODES), col_name])
        ):
            df_col.loc[
                (genstr(gen), fill, n_agents,
                 DIFFERENCE_ECBS_EN_MINUS_ICTS_EN),
                col_name
            ] = float(
                df_col.loc[
                    (genstr(gen), fill, n_agents, ECBS_EXPANDED_NODES),
                    col_name
                ] -
                df_col.loc[
                    (genstr(gen), fill, n_agents, ICTS_EXPANDED_NODES),
                    col_name
                ]
            )
        else:
            df_col.loc[
                (genstr(gen), fill, n_agents, ECBS_EXPANDED_NODES), col_name
            ] = np.nan
            df_col.loc[
                (genstr(gen), fill, n_agents, ICTS_EXPANDED_NODES), col_name
            ] = np.nan
    pb.end()
    return df_col


if __name__ == "__main__":
    main_icts()
