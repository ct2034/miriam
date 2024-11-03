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
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from definitions import INVALID, NO_SUCCESS, SUCCESS
from scenarios.evaluators import *
from scenarios.generators import *
from tools import BLUE_SEQ, RESET_SEQ, ProgressBar

GENERATOR = "generator"
FILL = "fill"
N_AGENTS = "n_agents"
# -------------------------
BRIDGES = "bridges"
CONNECTIVITY = "connectivity"
MEAN_DEGREE = "mean_degree"
N_EDGES = "n_edges"
N_EDGES_TA = "n_edges_ta"
N_NODES = "n_nodes"
N_NODES_TA = "n_nodes_ta"
SMALL_WORLD_OMEGA = "small_world_omega"
SMALL_WORLD_SIGMA = "small_world_sigma"
TREE_WIDTH = "tree_width"
UNCENTRALITY = "uncentrality"
WELL_FORMED = "well_formed"
# -------------------------
DIFF_INDEP = "diff_indep"
DIFF_SIM_DECEN_LEARNED = "diff_sim_decen_learned"
DIFF_SIM_DECEN_RANDOM = "diff_sim_decen_random"
DIFFERENCE_ECBS_EN_MINUS_ICTS_EN = "difference_ecbs_en_-_icts_en"
DIFFERENCE_SIM_DECEN_RADOM_MINUS_LEARNED = "difference_sim_decen_random_minus_learned"
ECBS_COST = "ecbs_cost"
ECBS_EDGE_BLOCKS = "ecbs_edge_blocks"
ECBS_EXPANDED_NODES = "ecbs_expanded_nodes"
ECBS_SUCCESS = "ecbs_success"
ECBS_VERTEX_BLOCKS = "ecbs_vertex_blocks"
ICTS_COST = "icts_cost"
ICTS_EXPANDED_NODES = "icts_expanded_nodes"
ICTS_SUCCESS = "icts_success"
SIM_DECEN_LEARNED_COST = "sim_decen_learned_cost"
SIM_DECEN_LEARNED_SUCCESS = "sim_decen_learned_success"
SIM_DECEN_RANDOM_COST = "sim_decen_random_cost"
SIM_DECEN_RANDOM_SUCCESS = "sim_decen_random_success"
USEFULLNESS = "usefullness"

# params for the generation of scenarios
scenario_params = [GENERATOR, FILL, N_AGENTS]

# evaluations from static analysis of the scenario or graph
evaluations_pre = [
    BRIDGES,
    CONNECTIVITY,
    MEAN_DEGREE,
    N_EDGES_TA,
    N_EDGES,
    N_NODES_TA,
    N_NODES,
    # SMALL_WORLD_OMEGA,
    # SMALL_WORLD_SIGMA,
    TREE_WIDTH,
    UNCENTRALITY,
    WELL_FORMED,
]

# evaluations after performing solving algorithms
evaluations_post = [
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
    SIM_DECEN_LEARNED_COST,
    SIM_DECEN_LEARNED_SUCCESS,
    SIM_DECEN_RANDOM_COST,
    SIM_DECEN_RANDOM_SUCCESS,
    # USEFULLNESS,
]


def init_values_debug():
    size = 8  # size for all scenarios
    n_fills = 3  # how many different fill values there should be
    n_n_agentss = 3  # how many different numbers of agents should there be"""
    n_runs = 8  # how many runs per configuration
    max_fill = 0.4  # maximal fill to sample until
    low_agents = 1  # lowest number of agents
    high_agents = 5  # highest number of agents
    return (max_fill, n_fills, n_n_agentss, n_runs, size, low_agents, high_agents)


def init_values_main():
    size = 8  # size for all scenarios
    n_fills = 8  # how many different fill values there should be
    n_n_agentss = 8  # how many different numbers of agents should there be"""
    n_runs = 512  # how many runs per configuration
    max_fill = 0.6  # maximal fill to sample until
    low_agents = 1  # lowest number of agents
    high_agents = 16  # highest number of agents
    return (max_fill, n_fills, n_n_agentss, n_runs, size, low_agents, high_agents)


def init_values_focus():
    size = 10  # size for all scenarios
    n_fills = 6  # how many different fill values there should be
    n_n_agentss = 6  # how many different numbers of agents should there be"""
    n_runs = 32  # how many runs per configuration
    max_fill = 0.7  # maximal fill to sample until
    low_agents = 5  # lowest number of agents
    high_agents = 10  # highest number of agents
    return (max_fill, n_fills, n_n_agentss, n_runs, size, low_agents, high_agents)


def get_fname(generator_name, evaluation, extension):
    return "scenarios/res_" + generator_name + "-" + evaluation + "." + extension


def genstr(generator):
    generator_name = (
        str(generator).split("at 0x")[0].replace("<function ", "").replace(" ", "")
    )
    return generator_name


def find_previous_values(experiment_matrix, fill, n_agents):
    i_f = experiment_matrix[FILL].index(fill)
    if i_f == 0:
        pf = None
    else:
        pf = experiment_matrix[FILL][i_f - 1]
    i_na = experiment_matrix[N_AGENTS].index(n_agents)
    if i_na == 0:
        pna = None
    else:
        pna = experiment_matrix[N_AGENTS][i_na - 1]
    return pf, pna


def make_full_df():
    # no warnings pls
    logging.getLogger("sim.decentralized.agent").setLevel(logging.ERROR)

    # (max_fill, n_fills, n_n_agentss, n_runs, size,
    #  low_agents, high_agents) = init_values_debug()
    (
        max_fill,
        n_fills,
        n_n_agentss,
        n_runs,
        size,
        low_agents,
        high_agents,
    ) = init_values_main()
    # (max_fill, n_fills, n_n_agentss, n_runs, size,
    #  low_agents, high_agents) = init_values_focus()

    # parameters to evaluate against
    generators = [random_fill, tracing_paths_in_the_dark, building_walls]

    fills = np.around(np.linspace(0, max_fill, n_fills), 2)  # list of fills we want

    n_agentss = np.linspace(
        low_agents, high_agents, n_n_agentss, dtype=int
    )  # list of different numbers of agents we want

    evaluations = evaluations_pre + evaluations_post

    experiment_matrix = {
        GENERATOR: generators,
        FILL: list(fills),
        N_AGENTS: list(n_agentss),
    }
    cols = evaluations + list(experiment_matrix.keys())
    df_results = pd.DataFrame(columns=cols)

    pbm = ProgressBar("main", n_runs)
    with Pool(4) as p:
        arguments = [(size, i_r, cols, experiment_matrix, pbm) for i_r in range(n_runs)]
        df_rowss = p.starmap(evaluate_full_run, arguments, chunksize=1)
    for df_rows in df_rowss:
        df_results = df_results.append(df_rows)
    pbm.end()
    df_results.info()
    df_results.to_pickle(get_fname("full", "orientation", "pkl"))
    return df_results


def evaluate_full_run(size, i_r, cols, experiment_matrix, pbm):
    # Taking care of some little pandas .......................................
    df_rows = pd.DataFrame(columns=cols)
    for gen, fill, n_agents in product(
        experiment_matrix[GENERATOR],
        experiment_matrix[FILL],
        experiment_matrix[N_AGENTS],
    ):
        # finding right row names .............................................
        row_name = f"seed{i_r}-{genstr(gen)}-fill{fill}-n_agents{n_agents}"
        previous_fill, previous_n_agents = find_previous_values(
            experiment_matrix, fill, n_agents
        )
        row_previous_fill = (
            f"seed{i_r}-{genstr(gen)}-fill{previous_fill}-n_agents{n_agents}"
        )
        row_previous_n_agents = (
            f"seed{i_r}-{genstr(gen)}-fill{fill}-n_agents{previous_n_agents}"
        )
        df_rows.loc[row_name] = [np.nan] * len(cols)
        # generating a scenario ...............................................
        env, starts, goals = gen(size, fill, n_agents, seed=i_r)
        df_rows.loc[row_name, GENERATOR] = genstr(gen)
        df_rows.loc[row_name, FILL] = fill
        df_rows.loc[row_name, N_AGENTS] = n_agents
        # graph based metrics .................................................
        if N_NODES in cols:
            df_rows.loc[row_name, N_NODES] = n_nodes(env)
        if N_EDGES in cols:
            df_rows.loc[row_name, N_EDGES] = n_edges(env)
        if N_NODES_TA in cols:
            df_rows.loc[row_name, N_NODES_TA] = n_nodes(env) * n_agents
        if N_EDGES_TA in cols:
            df_rows.loc[row_name, N_EDGES_TA] = n_edges(env) * n_agents
        if MEAN_DEGREE in cols:
            df_rows.loc[row_name, MEAN_DEGREE] = mean_degree(env)
        if TREE_WIDTH in cols:
            df_rows.loc[row_name, TREE_WIDTH] = tree_width(env)
        if SMALL_WORLD_OMEGA in cols:
            df_rows.loc[row_name, SMALL_WORLD_OMEGA] = small_world_omega(env)
        if SMALL_WORLD_SIGMA in cols:
            df_rows.loc[row_name, SMALL_WORLD_SIGMA] = small_world_sigma(env)
        if BRIDGES in cols:
            df_rows.loc[row_name, BRIDGES] = bridges(env)
        # static problem analysis .............................................
        if WELL_FORMED in cols:
            df_rows.loc[row_name, WELL_FORMED] = int(is_well_formed(env, starts, goals))
        if CONNECTIVITY in cols:
            df_rows.loc[row_name, CONNECTIVITY] = connectivity(env, starts, goals)
        if UNCENTRALITY in cols:
            df_rows.loc[row_name, UNCENTRALITY] = uncentrality(env, starts, goals)
        # calculating ecbs cost ...............................................
        if ECBS_SUCCESS in cols:
            if (
                previous_fill is not None
                and df_rows.loc[row_previous_fill, ECBS_SUCCESS] == NO_SUCCESS
            ):
                # previous fills timed out
                res_ecbs = INVALID
            elif (
                previous_n_agents is not None
                and df_rows.loc[row_previous_n_agents, ECBS_SUCCESS] == NO_SUCCESS
            ):
                # previous agent count failed as well
                res_ecbs = INVALID
            else:
                res_ecbs = cost_ecbs(env, starts, goals)
            if res_ecbs == INVALID:
                df_rows.loc[row_name, ECBS_SUCCESS] = NO_SUCCESS
            else:  # valid ecbs result
                df_rows.loc[row_name, ECBS_SUCCESS] = SUCCESS
                if ECBS_COST in cols:
                    df_rows.loc[row_name, ECBS_COST] = res_ecbs
                if ECBS_EXPANDED_NODES in cols:
                    df_rows.loc[row_name, ECBS_EXPANDED_NODES] = expanded_nodes_ecbs(
                        env, starts, goals
                    )
                    # evaluating blocks
                    blocks = blocks_ecbs(env, starts, goals)
                    if blocks != INVALID and ECBS_VERTEX_BLOCKS in cols:
                        df_rows.loc[row_name, ECBS_VERTEX_BLOCKS] = blocks[0]
                    if blocks != INVALID and ECBS_EDGE_BLOCKS in cols:
                        df_rows.loc[row_name, ECBS_EDGE_BLOCKS] = blocks[1]
        # what is icts cost? ..................................................
        if ICTS_SUCCESS in cols:
            if (
                previous_fill is not None
                and df_rows.loc[row_previous_fill, ICTS_SUCCESS] == NO_SUCCESS
            ):
                # previous fills timed out
                res_icts = INVALID
            elif (
                previous_n_agents is not None
                and df_rows.loc[row_previous_n_agents, ICTS_SUCCESS] == NO_SUCCESS
            ):
                # previous agent count failed as well
                res_icts = INVALID
            else:
                res_icts = cost_icts(env, starts, goals)
            if res_icts == INVALID:
                df_rows.loc[row_name, ICTS_SUCCESS] = NO_SUCCESS
            else:
                df_rows.loc[row_name, ICTS_SUCCESS] = SUCCESS
                if ICTS_COST in cols:
                    df_rows.loc[row_name, ICTS_COST] = res_icts
                if ICTS_EXPANDED_NODES in cols:
                    df_rows.loc[row_name, ICTS_EXPANDED_NODES] = expanded_nodes_icts(
                        env, starts, goals
                    )
        # run icts and compare n of expanded nodes
        if (
            DIFFERENCE_ECBS_EN_MINUS_ICTS_EN in cols
            and not np.isnan(df_rows.loc[row_name, ECBS_EXPANDED_NODES])
            and not np.isnan(df_rows.loc[row_name, ICTS_EXPANDED_NODES])
        ):
            df_rows.loc[row_name, DIFFERENCE_ECBS_EN_MINUS_ICTS_EN] = float(
                df_rows.loc[row_name, ECBS_EXPANDED_NODES]
                - df_rows.loc[row_name, ICTS_EXPANDED_NODES]
            )
        # decentralized sim ###################################################
        if SIM_DECEN_RANDOM_SUCCESS in cols:
            decen_cost_r = cost_sim_decentralized_random(env, starts, goals)
            df_rows.loc[row_name, SIM_DECEN_RANDOM_SUCCESS] = int(
                decen_cost_r != INVALID
            )
            if SIM_DECEN_RANDOM_COST in cols and decen_cost_r != INVALID:
                df_rows.loc[row_name, SIM_DECEN_RANDOM_COST] = decen_cost_r
        if SIM_DECEN_LEARNED_SUCCESS in cols:
            decen_cost_l = cost_sim_decentralized_learned(env, starts, goals)
            df_rows.loc[row_name, SIM_DECEN_LEARNED_SUCCESS] = int(
                decen_cost_l != INVALID
            )
            if SIM_DECEN_LEARNED_COST in cols and decen_cost_l != INVALID:
                df_rows.loc[row_name, SIM_DECEN_LEARNED_COST] = decen_cost_l
        # comparing decentralized to optimal solution #########################
        if df_rows.loc[row_name, ECBS_SUCCESS] == SUCCESS:
            ecbs_cost = df_rows.loc[row_name, ECBS_COST]
            # indep ...........................................................
            if DIFF_INDEP in cols:
                indep_cost = cost_independent(env, starts, goals)
                if indep_cost != INVALID:
                    df_rows.loc[row_name, DIFF_INDEP] = ecbs_cost - indep_cost
            decen_cost_r = df_rows.loc[row_name, SIM_DECEN_RANDOM_COST]
            decen_cost_l = df_rows.loc[row_name, SIM_DECEN_LEARNED_COST]
            # decen random ....................................................
            if DIFF_SIM_DECEN_RANDOM in cols and decen_cost_r != INVALID:
                df_rows.loc[row_name, DIFF_SIM_DECEN_RANDOM] = decen_cost_r - ecbs_cost
            # decen learned ...................................................
            if DIFF_SIM_DECEN_LEARNED in cols and decen_cost_l != INVALID:
                df_rows.loc[row_name, DIFF_SIM_DECEN_LEARNED] = decen_cost_l - ecbs_cost
            # diff random minus learned .......................................
            if (
                DIFFERENCE_SIM_DECEN_RADOM_MINUS_LEARNED in cols
                and decen_cost_r != INVALID
                and decen_cost_l != INVALID
            ):
                df_rows.loc[row_name, DIFFERENCE_SIM_DECEN_RADOM_MINUS_LEARNED] = (
                    decen_cost_r - decen_cost_l
                )
    pbm.progress(i_r)
    return df_rows


if __name__ == "__main__":
    logging.getLogger("sim.decentralized.policy").setLevel(logging.ERROR)
    df_results = make_full_df()
