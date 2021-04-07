#!/usr/bin/env python3
import argparse
import logging
import pickle
from collections import OrderedDict
from copy import copy
from functools import lru_cache
from itertools import product, repeat
from math import ceil
from multiprocessing import Pool
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
DIFF_INDEP = "diff_indep"
DIFF_SIM_DECEN_LEARNED = "diff_sim_decen_learned"
DIFF_SIM_DECEN_RANDOM = "diff_sim_decen_random"
DIFFERENCE_ECBS_EN_MINUS_ICTS_EN = "difference_ecbs_en_-_icts_en"
DIFFERENCE_SIM_DECEN_RADOM_MINUS_LEARNED = "difference_sim_decen_minus_learned"
ECBS_COST = "ecbs_cost"
ECBS_EDGE_BLOCKS = "ecbs_edge_blocks"
ECBS_EXPANDED_NODES = "ecbs_expanded_nodes"
ECBS_SUCCESS = "ecbs_success"
ECBS_VERTEX_BLOCKS = "ecbs_vertex_blocks"
ICTS_COST = "icts_cost"
ICTS_EXPANDED_NODES = "icts_expanded_nodes"
ICTS_SUCCESS = "icts_success"
MEAN_DEGREE = "mean_degree"
N_EDGES = "n_edges"
N_EDGES_TA = "n_edges_ta"
N_NODES = "n_nodes"
N_NODES_TA = "n_nodes_ta"
SIM_DECEN_LEARNED_COST = "sim_decen_learned_cost"
SIM_DECEN_LEARNED_SUCCESS = "sim_decen_learned_success"
SIM_DECEN_RANDOM_COST = "sim_decen_random_cost"
SIM_DECEN_RANDOM_SUCCESS = "sim_decen_random_success"
USEFULLNESS = "usefullness"
WELL_FORMED = "well_formed"


def init_values_debug():
    size = 8  # size for all scenarios
    n_fills = 3  # how many different fill values there should be
    n_n_agentss = 3  # how many different numbers of agents should there be"""
    n_runs = 8  # how many runs per configuration
    max_fill = .4  # maximal fill to sample until
    low_agents = 1  # lowest number of agents
    high_agents = 5  # highest number of agents
    return (max_fill, n_fills, n_n_agentss, n_runs, size,
            low_agents, high_agents)


def init_values_main():
    size = 8  # size for all scenarios
    n_fills = 8  # how many different fill values there should be
    n_n_agentss = 8  # how many different numbers of agents should there be"""
    n_runs = 128  # how many runs per configuration
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
        df: pd.DataFrame, generator_name: str,
        title: str, normalize_cbars_for: Optional[str] = None,
        success_required: float = .05):
    evaluations = sorted(list(set(df.index.get_level_values('evaluations'))))
    n_agentss = sorted(list(set(df.index.get_level_values('n_agentss'))))
    n_n_agentss = len(n_agentss)
    fills = sorted(list(set(df.index.get_level_values('fills'))))
    n_fills = len(fills)

    # our cmap with support for over / under
    palette = copy(plt.cm.PuOr)
    palette.set_over('b', 1.0)
    palette.set_under('r', 1.0)
    palette.set_bad('k', 1.0)

    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(title+generator_name, fontsize=16)
    subplot_rows: int = 2
    subplot_cols: int = int(np.ceil(len(evaluations) / 2))
    minmax: Tuple[Optional[float], Optional[float]] = (None, None)

    if normalize_cbars_for is not None:
        # same min / max for top bottom plots
        mins = 99999
        maxs = 0
        for i, ev in enumerate(evaluations):
            if normalize_cbars_for in ev:
                this_ev_data = df.xs(ev, level=EVALUATIONS)
                for i_f, i_a in product(range(n_fills),
                                        range(n_n_agentss)):
                    fill = fills[i_f]
                    n_agents = n_agentss[i_a]
                    this_data = this_ev_data.loc[(
                        generator_name, fill, n_agents)].to_numpy()
                    if len(this_data[np.logical_not(np.isnan(this_data))]) > 0:
                        this_mean = np.mean(
                            this_data[np.logical_not(np.isnan(this_data))])
                        mins = min(this_mean, mins)
                        maxs = max(this_mean, maxs)
        minmax = (-1*max(abs(mins), abs(maxs)), max(abs(mins), abs(maxs)))

    for i, ev in enumerate(evaluations):
        if normalize_cbars_for in ev:
            this_minmax = minmax
        else:
            this_minmax = (None, None)
        this_ev_data = df.xs(ev, level=EVALUATIONS)
        r_final = np.full([n_fills, n_n_agentss], np.nan, dtype=float)
        for i_f, i_a in product(range(n_fills),
                                range(n_n_agentss)):
            fill = fills[i_f]
            n_agents = n_agentss[i_a]
            this_data = this_ev_data.loc[(
                generator_name, fill, n_agents)].to_numpy()
            n_required = ceil(len(this_data) * success_required)
            if len(this_data[np.logical_not(np.isnan(this_data))]
                   ) >= n_required:
                r_final[i_f, i_a] = np.mean(
                    this_data[np.logical_not(np.isnan(this_data))]
                )
        ax = fig.add_subplot(subplot_rows, subplot_cols, i+1)
        im = ax.imshow(
            r_final,
            cmap=palette,
            vmin=this_minmax[0],
            vmax=this_minmax[1],
            origin='lower'
        )
        fig.colorbar(im, extend='both', spacing='uniform',
                     shrink=0.9, ax=ax)
        plt.title(evaluations[i])
        plt.ylabel('Fills')
        plt.yticks(range(n_fills), map(lambda a: str(a), fills))
        plt.xlabel('Agents')
        plt.xticks(range(n_n_agentss), map(
            lambda a: str(int(a)), n_agentss))

    plt.tight_layout()
    fname = get_fname(title, generator_name, "png")
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


def make_full_df():
    # no warnings pls
    logging.getLogger('sim.decentralized.agent').setLevel(logging.ERROR)

    # (max_fill, n_fills, n_n_agentss, n_runs, size,
    #  low_agents, high_agents) = init_values_debug()
    (max_fill, n_fills, n_n_agentss, n_runs, size,
     low_agents, high_agents) = init_values_main()
    # (max_fill, n_fills, n_n_agentss, n_runs, size,
    #  low_agents, high_agents) = init_values_focus()

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
        DIFF_INDEP,
        DIFF_SIM_DECEN_LEARNED,
        DIFF_SIM_DECEN_RANDOM,
        DIFFERENCE_SIM_DECEN_RADOM_MINUS_LEARNED,
        # DIFFERENCE_ECBS_EN_MINUS_ICTS_EN,
        ECBS_COST,
        # ECBS_EDGE_BLOCKS,
        # ECBS_EXPANDED_NODES,
        ECBS_SUCCESS,
        # ECBS_VERTEX_BLOCKS,
        ICTS_COST,
        # ICTS_EXPANDED_NODES,
        ICTS_SUCCESS,
        MEAN_DEGREE,
        N_EDGES_TA,
        N_EDGES,
        N_NODES_TA,
        N_NODES,
        SIM_DECEN_LEARNED_COST,
        SIM_DECEN_LEARNED_SUCCESS,
        SIM_DECEN_RANDOM_COST,
        SIM_DECEN_RANDOM_SUCCESS,
        # USEFULLNESS,
        WELL_FORMED
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

    pbm = ProgressBar("main", 0)  # timing only
    with Pool(4) as p:
        arguments = [(i_r, idx, generators, fills, n_agentss, size)
                     for i_r in range(n_runs)]
        df_cols = p.starmap(evaluate_full, arguments, chunksize=1)
    for df_col in df_cols:
        add_colums(df_results, df_col)
    pbm.end()

    # with pd.option_context('display.max_rows',
    #                        None,
    #                        'display.max_columns',
    #                        None):  # all rows and columns
    #     print(df_results)
    print(df_results.info)

    df_results.to_pickle(get_fname("full", "_", "pkl"))
    return df_results


def evaluate_full(i_r, idx, generators, fills, n_agentss, size):
    col_name = "seed{}".format(i_r)
    pb = ProgressBar("column >{}<".format(col_name),
                     (len(generators) * len(fills) * len(n_agentss)), 50)
    # Taking care of some little pandas ...........................
    df_col = pd.DataFrame(index=idx, dtype=float)
    df_col[col_name] = [np.nan] * len(df_col.index)
    df_col.sort_index(inplace=True)
    evaluations = idx.get_level_values('evaluations')
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
        if N_NODES in evaluations:
            df_col.loc[(genstr(gen), fill, n_agents, N_NODES), col_name
                       ] = n_nodes(env)
        if N_EDGES in evaluations:
            df_col.loc[(genstr(gen), fill, n_agents, N_EDGES), col_name
                       ] = n_edges(env)
        if N_NODES_TA in evaluations:
            df_col.loc[(genstr(gen), fill, n_agents, N_NODES_TA), col_name
                       ] = n_nodes(env) * n_agents
        if N_EDGES_TA in evaluations:
            df_col.loc[(genstr(gen), fill, n_agents, N_EDGES_TA), col_name
                       ] = n_edges(env) * n_agents
        if MEAN_DEGREE in evaluations:
            df_col.loc[(genstr(gen), fill, n_agents, MEAN_DEGREE), col_name
                       ] = mean_degree(env)
        # well-formedness .....................................................
        if WELL_FORMED in evaluations:
            df_col.loc[
                (genstr(gen), fill,
                 n_agents, WELL_FORMED), col_name
            ] = int(is_well_formed(env, starts, goals))
        # calculating ecbs cost ...............................................
        if ECBS_SUCCESS in evaluations:
            if i_f > 0 and df_col.loc[
                    (genstr(gen), fills[i_f-1],
                     n_agents, ECBS_SUCCESS), col_name
            ] == NO_SUCCESS:
                # previous fills timed out
                res_ecbs = INVALID
            elif i_a > 0 and df_col.loc[
                    (genstr(gen), fill,
                     n_agentss[i_a-1], ECBS_SUCCESS), col_name
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
                if ECBS_COST in evaluations:
                    df_col.loc[
                        (genstr(gen), fill, n_agents, ECBS_COST),
                        col_name] = res_ecbs
                if ECBS_EXPANDED_NODES in evaluations:
                    df_col.loc[
                        (genstr(gen), fill, n_agents,
                         ECBS_EXPANDED_NODES), col_name
                    ] = expanded_nodes_ecbs(
                        env, starts, goals)
                    # evaluating blocks
                    blocks = blocks_ecbs(env, starts, goals)
                    if blocks != INVALID and ECBS_VERTEX_BLOCKS in evaluations:
                        df_col.loc[
                            (genstr(gen), fill, n_agents,
                             ECBS_VERTEX_BLOCKS), col_name] = blocks[0]
                    if blocks != INVALID and ECBS_EDGE_BLOCKS in evaluations:
                        df_col.loc[
                            (genstr(gen), fill, n_agents,
                             ECBS_EDGE_BLOCKS), col_name] = blocks[1]
        # what is icts cost? ..................................................
        if ICTS_SUCCESS in evaluations:
            if i_f > 0 and df_col.loc[
                    (genstr(gen), fills[i_f-1],
                     n_agents, ICTS_SUCCESS), col_name
            ] == NO_SUCCESS:
                # previous fills timed out
                res_icts = INVALID
            elif i_a > 0 and df_col.loc[
                    (genstr(gen), fill,
                     n_agentss[i_a-1], ICTS_SUCCESS), col_name
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
                if ICTS_COST in evaluations:
                    df_col.loc[
                        (genstr(gen), fill, n_agents, ICTS_COST),
                        col_name] = res_icts
                if ICTS_EXPANDED_NODES in evaluations:
                    df_col.loc[
                        (genstr(gen), fill, n_agents,
                         ICTS_EXPANDED_NODES), col_name
                    ] = expanded_nodes_icts(
                        env, starts, goals)
        # run icts and compare n of expanded nodes
        if DIFFERENCE_ECBS_EN_MINUS_ICTS_EN in evaluations:
            if (
                not np.isnan(df_col.loc[
                    (genstr(gen), fill, n_agents, ECBS_EXPANDED_NODES),
                    col_name])
                and
                not np.isnan(df_col.loc[
                    (genstr(gen), fill, n_agents, ICTS_EXPANDED_NODES),
                    col_name])
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
        # comparing costs to optimal solution #################################
        if df_col.loc[
            (genstr(gen), fill,
             n_agentss[i_a], ECBS_SUCCESS), col_name] == SUCCESS:
            ecbs_cost = df_col.loc[
                (genstr(gen), fill,
                 n_agentss[i_a], ECBS_COST), col_name]
            if DIFF_INDEP in evaluations:
                indep_cost = cost_independent(env, starts, goals)
                if indep_cost != INVALID:
                    df_col.loc[
                        (genstr(gen), fill, n_agentss[i_a], DIFF_INDEP),
                        col_name] = ecbs_cost - indep_cost
            if DIFF_SIM_DECEN_RANDOM in evaluations:
                decen_cost_r = cost_sim_decentralized_random(
                    env, starts, goals)
                if decen_cost_r != INVALID:
                    df_col.loc[
                        (genstr(gen), fill,
                         n_agentss[i_a], DIFF_SIM_DECEN_RANDOM),
                        col_name] = decen_cost_r - ecbs_cost
                    # if decen_cost < ecbs_cost:
                    #     print(
                    #         '~~ decen_cost < ecbs_cost for ... ~~~' +
                    #         f'env: {str(env)}\nstarts: {str(starts)}\n' +
                    #         f'goals: {str(goals)}\n' +
                    #         f'~~~~~~~~~~~~\n' +
                    #         f'{repr((env, starts, goals))}\n' +
                    #         f'~~~~~~~~~~~~')
                    # TODO: It is true, that sometimes, decentralized
                    # solutions have lower cost than ecbs, which seems to
                    # be down to decentralized solutions ignoring finished
                    # agents.
                if SIM_DECEN_RANDOM_SUCCESS in evaluations:
                    df_col.loc[
                        (genstr(gen), fill,
                         n_agentss[i_a], SIM_DECEN_RANDOM_SUCCESS),
                        col_name] = int(decen_cost_r != INVALID)
                if (SIM_DECEN_RANDOM_COST in evaluations and
                        decen_cost_r != INVALID):
                    df_col.loc[
                        (genstr(gen), fill,
                         n_agentss[i_a], SIM_DECEN_RANDOM_COST),
                        col_name] = decen_cost_r
            if DIFF_SIM_DECEN_LEARNED in evaluations:
                decen_cost_l = cost_sim_decentralized_learned(
                    env, starts, goals)
                if decen_cost_l != INVALID:
                    df_col.loc[
                        (genstr(gen), fill,
                         n_agentss[i_a], DIFF_SIM_DECEN_LEARNED),
                        col_name] = decen_cost_l - ecbs_cost
                if SIM_DECEN_LEARNED_SUCCESS in evaluations:
                    df_col.loc[
                        (genstr(gen), fill,
                         n_agentss[i_a], SIM_DECEN_LEARNED_SUCCESS),
                        col_name] = int(decen_cost_l != INVALID)
                if (SIM_DECEN_LEARNED_COST in evaluations and
                        decen_cost_l != INVALID):
                    df_col.loc[
                        (genstr(gen), fill,
                         n_agentss[i_a], SIM_DECEN_LEARNED_COST),
                        col_name] = decen_cost_l
            if (DIFFERENCE_SIM_DECEN_RADOM_MINUS_LEARNED in evaluations
                    and decen_cost_r != INVALID and decen_cost_l != INVALID):
                df_col.loc[
                    (genstr(gen), fill,
                     n_agentss[i_a],
                     DIFFERENCE_SIM_DECEN_RADOM_MINUS_LEARNED),
                    col_name] = decen_cost_r - decen_cost_l

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
    parser = argparse.ArgumentParser()
    parser.add_argument("pkl_read", nargs='?')
    args = parser.parse_args()
    print(f'args.pkl_read: {args.pkl_read}')
    if args.pkl_read:
        df_results = pd.read_pickle(args.pkl_read)
    else:
        df_results = make_full_df()

    for gen in sorted(list(set(
        df_results.index.get_level_values(GENERATORS)
    ))):
        plot_images(
            df_results,
            title="full",
            generator_name=gen,
            normalize_cbars_for="diff"
        )

    # compare expanded nodes over graph properties
    # xs = [N_NODES, N_EDGES, MEAN_DEGREE, N_NODES_TA, N_EDGES_TA]
    # ys = [DIFFERENCE_ECBS_EN_MINUS_ICTS_EN]
    # plot_scatter(df_results, xs, ys, cs=GENERATORS, titles=xs)

    plt.show()
