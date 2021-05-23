#!/usr/bin/env python3
import argparse
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from scenarios.main_eval import (evaluations_post, evaluations_pre,
                                 scenario_params)


def sort_cols_based_on_pre_post(cols):
    cols_scen = []
    cols_pre = []
    cols_post = []
    for c in cols:
        if c in scenario_params:
            cols_scen.append(c)
        elif c in evaluations_pre:
            cols_pre.append(c)
        elif c in evaluations_post:
            cols_post.append(c)
        else:
            assert False, "column must be in a list"
    assert (len(cols) == len(cols_scen) + len(cols_pre) +
            len(cols_post)), "All columns must end up somewhere"
    cols_scen.sort()
    cols_pre.sort()
    cols_post.sort()
    return cols_scen + cols_pre + cols_post


if __name__ == "__main__":
    """Load pandas results from pkl file and perform a svd on the data"""
    parser = argparse.ArgumentParser()
    parser.add_argument("pkl_read")
    args = parser.parse_args()
    print(f'args.pkl_read: {args.pkl_read}')
    df_results: pd.DataFrame = pd.read_pickle(args.pkl_read)
    print(df_results.head())
    print(df_results.shape)

    ignored_evaluations: List[str] = [
        # "n_edges",
        # "n_edges_ta",
        # "n_nodes",
        # "n_nodes_ta"
    ]
    generators = list(set(df_results["generator"]))

    # changing dataframe ######################################################
    df_results.sort_index(inplace=True)
    cols = list(df_results.columns)
    cols = sort_cols_based_on_pre_post(cols)
    print(cols)
    text_cols = ["generator"]
    for c in cols:
        if c not in text_cols:
            df_results[c] = df_results[c].astype(float)
    df_results = df_results[df_results.columns[list(map(
        lambda c: list(df_results.columns).index(c), cols
    ))]]

    # dropping ignored rows ###################################################
    df_results_clean = df_results.drop(labels=ignored_evaluations, axis=1)
    for ev in ignored_evaluations:
        cols.pop(cols.index(ev))

    # preparation #############################################################
    print("Dropping any nan rows ...")
    df_results_clean.dropna(axis=0, how='any', inplace=True)
    # print(df_results_clean.head())
    print(df_results_clean.shape)
    # sns.heatmap(df_results_clean)
    # plt.show()

    first_std = df_results_clean.std()
    for c in cols:
        if c in first_std.index and first_std[c] == 0:
            df_results_clean.drop(columns=[c], inplace=True)
        elif c not in first_std.index:  # non numeric column
            df_results_clean.drop(columns=[c], inplace=True)
    df_results_clean = df_results_clean.apply(pd.to_numeric)
    numeric_var_cols = df_results_clean.columns
    print(numeric_var_cols)
    mean = df_results_clean.mean()
    std = df_results_clean.std()
    df_results_scaled = (df_results_clean-mean) / std
    print(df_results_scaled.head())

    # svd #####################################################################
    U, S, V = np.linalg.svd(df_results_scaled.transpose(), full_matrices=False)

    # analyzing explained variance ############################################
    var_explained = S/np.sum(S)
    var_explained = var_explained[:len(cols)]

    plt.figure()
    sns.barplot(x=list(range(1, len(var_explained)+1)),
                y=var_explained)
    plt.xlabel('SVs', fontsize=16)
    plt.ylabel('Percent Variance Explained', fontsize=16)
    plt.savefig("scenarios/plots/variance_explained.png")

    # visualizing matrices ####################################################
    plt.figure()
    print(f"U.shape {U.shape}")
    sns.heatmap(U[:len(numeric_var_cols), :], cmap="RdBu")
    plt.title("U")
    plt.yticks(
        list(map(lambda x: x+.5, range(len(numeric_var_cols)))),
        numeric_var_cols,
        rotation='horizontal')
    plt.savefig("scenarios/plots/heatmap_u.png")

    plt.figure()
    sns.heatmap(V, cmap="RdBu")
    plt.title("V")
    plt.savefig("scenarios/plots/heatmap_v.png")

    # what a (over)view #######################################################
    for g in generators:
        sns.pairplot(
            df_results[df_results["generator"] == g],
            # hue="generator",
            # markers=["o", "s", "D"],
            kind="kde",
            # plot_kws={"s": 3},
            x_vars=[
                "mean_degree",
                "n_nodes_ta",
                "n_edges_ta",
                "bridges",
                "connectivity",
                "tree_width",
                "uncentrality"
            ],
            y_vars=[
                "ecbs_cost",
                "icts_cost",
                "diff_indep",
                "diff_sim_decen_learned",
                "difference_sim_decen_random_minus_learned"
            ],
        )
        plt.savefig(f"scenarios/plots/pairplot-{g}.png", dpi=500)

    # is there something? #####################################################
    plt.figure()
    sns.set_theme(color_codes=True)
    sns.lmplot(
        x="mean_degree",
        y="ecbs_cost",
        hue="generator",
        markers=["o", "s", "D"],
        # hue='ecbs_cost',
        data=df_results,
        palette="viridis",
        scatter_kws={"s": 3}
    )
    plt.savefig(
        "scenarios/plots/mean_degree-ecbs_cost.png", dpi=500)

    plt.figure()
    sns.set_theme(color_codes=True)
    sns.displot(
        kind="kde",
        # sns.lmplot(
        x="uncentrality",
        y="diff_indep",
        # hue="generator",
        # markers=["o", "s", "D"],
        data=df_results,
        palette="viridis",
        # scatter_kws={"s": 3}
    )
    plt.savefig(
        "scenarios/plots/uncentrality-diff_indep.png", dpi=500)

    # plt.figure()
    # sns.scatterplot(
    #     x='mean_degree',
    #     y='diff_indep',
    #     hue='ecbs_cost',
    #     data=df_results,
    #     palette="viridis",
    #     s=10)

    # plt.figure()
    # sns.scatterplot(
    #     x='n_agents',
    #     y='diff_indep',
    #     hue='mean_degree',
    #     data=df_results,
    #     palette="viridis",
    #     s=10)

    # plt.figure()
    # sns.scatterplot(
    #     x='mean_degree',
    #     y='diff_sim_decen_learned',
    #     hue='n_agents',
    #     data=df_results,
    #     palette="viridis",
    #     s=10)

    # plt.show()
