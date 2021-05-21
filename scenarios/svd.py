#!/usr/bin/env python3
import argparse
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

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

    # changing dataframe ######################################################
    df_results.sort_index(inplace=True)
    cols = list(df_results.columns)
    cols.sort()
    print(cols)

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
    U, S, V = np.linalg.svd(df_results_scaled, full_matrices=False)

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
    sns.pairplot(
        df_results,
        hue="generator",
        markers=["o", "s", "D"],
        plot_kws={"s": 3},
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
    plt.savefig("scenarios/plots/pairplot.png", dpi=500)

    # is there something? #####################################################
    # plt.figure()
    # sns.scatterplot(
    #     x='mean_degree',
    #     y='fill',
    #     hue='ecbs_cost',
    #     data=df_results,
    #     palette="viridis",
    #     s=10)

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
