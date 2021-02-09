#!/usr/bin/env python3
import logging
import pickle
from collections import OrderedDict
from copy import copy
from functools import lru_cache
from itertools import product
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


def init_values_debug():
    size = 8  # size for all scenarios
    n_fills = 4  # how many different fill values there should be
    n_n_agentss = 4  # how many different numbers of agents should there be"""
    n_runs = 4  # how many runs per configuration
    max_fill = .4  # maximal fill to sample until
    return max_fill, n_fills, n_n_agentss, n_runs, size


def init_values_main():
    size = 8  # size for all scenarios
    n_fills = 8  # how many different fill values there should be
    n_n_agentss = 8  # how many different numbers of agents should there be"""
    n_runs = 32  # how many runs per configuration
    max_fill = .6  # maximal fill to sample until
    return max_fill, n_fills, n_n_agentss, n_runs, size


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


def plot_results(
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
        r_final = np.full([n_fills, n_n_agentss], np.nan, dtype=np.float)
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


def main_icts():
    # no warnings pls
    logging.getLogger('sim.decentralized.agent').setLevel(logging.ERROR)

    # max_fill, n_fills, n_n_agentss, n_runs, size = init_values_debug()
    max_fill, n_fills, n_n_agentss, n_runs, size = init_values_main()

    # parameters to evaluate against
    generators = [
        # like_policylearn_gen,
        like_sim_decentralized,
        tracing_pathes_in_the_dark
    ]

    fills = np.around(
        np.linspace(0, max_fill, n_fills),
        2
    )  # list of fills we want

    n_agentss = np.linspace(1, 16, n_n_agentss, dtype=np.int
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
        DIFFERENCE_ECBS_EN_MINUS_ICTS_EN
    ]

    # preparing panda dataframes
    index_arrays = {
        'generators': list(map(genstr, generators)),
        'fills': fills,
        'n_agentss': n_agentss,
        EVALUATIONS: evaluations
    }
    idx = pd.MultiIndex.from_product(
        index_arrays.values(), names=index_arrays.keys())
    df_results = pd.DataFrame(index=idx)

    df_results.sort_index(inplace=True)
    assert len(index_arrays) == df_results.index.lexsort_depth

    pdo = ProgressBar(BLUE_SEQ + "Overall" + RESET_SEQ, n_runs)
    for i_r in range(n_runs):
        df_col = evaluate_one_column(
            i_r, idx, generators, fills, n_agentss, size)
        add_colums(df_results, df_col)
        pdo.progress()
    pdo.end()

    with pd.option_context('display.max_rows',
                           None,
                           'display.max_columns',
                           None):  # all rows and columns
        print(df_results)
    print(df_results.info)

    df_results.to_pickle(get_fname_both("icts", "pkl"))

    data_to_print = OrderedDict()
    for gen in generators:
        genname = genstr(gen)
        # data_to_print[genname+" ECBS success"] = df_results.loc[
        #     (genname)].xs(ECBS_SUCCESS, level=EVALUATIONS)
        # data_to_print[genname+"..."] = df_results.loc[
        #     (genname)].xs(ECBS_COST, level=EVALUATIONS)
        data_to_print[genname+" ECBS expanded nodes"] = df_results.loc[
            (genname)].xs(ECBS_EXPANDED_NODES, level=EVALUATIONS)
        # data_to_print[genname+"..."] = df_results.loc[
        #     (genname)].xs(ECBS_VERTEX_BLOCKS, level=EVALUATIONS)
        # data_to_print[genname+"..."] = df_results.loc[
        #     (genname)].xs(ECBS_EDGE_BLOCKS, level=EVALUATIONS)
        # data_to_print[genname+" ICTS success"] = df_results.loc[
        #     (genname)].xs(ICTS_SUCCESS, level=EVALUATIONS)
        # data_to_print[genname+"..."] = df_results.loc[
        #     (genname)].xs(ICTS_COST, level=EVALUATIONS)
        data_to_print[genname+" ICTS expanded nodes"] = df_results.loc[
            (genname)].xs(ICTS_EXPANDED_NODES, level=EVALUATIONS)
        data_to_print[genname+" Difference ECBS minus ICTS expanded nodes"
                      ] = df_results.loc[(genname)].xs(
                          DIFFERENCE_ECBS_EN_MINUS_ICTS_EN,
            level=EVALUATIONS)
        # plot
    plot_results(
        list(data_to_print.values()),
        list(data_to_print.keys()),
        title="expanded-nodes",
        n_agentss=n_agentss,
        fills=fills,
        evaluation="icts",
        normalize_cbars="expanded"
    )
    plt.show()


def evaluate_one_column(i_r, idx, generators, fills, n_agentss, size):
    col_name = "seed{}".format(i_r)
    pb = ProgressBar("column >{}<".format(col_name),
                     (len(generators) * len(fills) * len(n_agentss)), 10)
    # Taking care of some little pandas ...........................
    df_col = pd.DataFrame(index=idx)
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
            df_col.loc[
                (genstr(gen), fill, n_agents, ECBS_EXPANDED_NODES), col_name
            ] is not np.nan and
            df_col.loc[
                (genstr(gen), fill, n_agents, ICTS_EXPANDED_NODES), col_name
            ] is not np.nan
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


if __name__ == "__main__":
    main_icts()
