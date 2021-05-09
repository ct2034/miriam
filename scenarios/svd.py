#!/usr/bin/env python3
import argparse

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from numpy.core.fromnumeric import var
from tools import ProgressBar

if __name__ == "__main__":
    """Load pandas results from pkl file and perform a svd on the data"""
    parser = argparse.ArgumentParser()
    parser.add_argument("pkl_read")
    args = parser.parse_args()
    print(f'args.pkl_read: {args.pkl_read}')
    df_results: pd.DataFrame = pd.read_pickle(args.pkl_read)
    print(df_results.head())
    print(df_results.shape)

    ignored_evaluations = [
        "n_edges",
        "n_edges_ta",
        "n_nodes",
        "n_nodes_ta"
    ]

    # changing dataframe ######################################################
    if "svd" in args.pkl_read:  # was already transformed
        df_svd = df_results
        df_svd.sort_index(inplace=True)
        new_idx = list(set(df_svd.index))
        new_idx.sort()
        print(new_idx)
    else:  # transforming dataframe into more columns
        idx = df_results.index
        print(idx.names)
        new_idx = list(set(idx.get_level_values(
            idx.names[-1])))
        print(new_idx)
        new_idx += [idx.names[1]]  # fills
        new_idx += [idx.names[2]]  # n_agents
        print(new_idx)
        df_svd = pd.DataFrame(index=new_idx)
        print(df_svd.head())
        idx.sort_values()

        pb = ProgressBar("columns", len(df_results.columns), 10)
        for col in df_results.columns:
            for i in idx:
                # print(i)
                new_col = col+"_"+"_".join(map(str, i[:-1]))
                # print(new_col)
                df_svd.loc[i[-1], new_col] = df_results.loc[i, col]
                df_svd.loc[idx.names[1], new_col] = i[1]  # fills
                df_svd.loc[idx.names[2], new_col] = i[2]  # n_agents
            pb.progress()
        pb.end()
        print(df_svd.head())
        print(df_svd.shape)
        df_svd.to_pickle("svd_"+args.pkl_read)

    # sns.heatmap(df_svd)
    # plt.show()

    # dropping ignored rows ###################################################
    df_svd.drop(labels=ignored_evaluations, axis=0, inplace=True)
    for ev in ignored_evaluations:
        new_idx.pop(new_idx.index(ev))

    # preparation #############################################################
    print("Dropping all nan rows ...")
    df_svd.dropna(axis=0, how='all', inplace=True)
    # print(df_svd.head())
    print(df_svd.shape)
    # sns.heatmap(df_svd)
    # plt.show()

    print("Dropping any nan cols ...")
    df_svd.dropna(axis=1, how='any', inplace=True)
    # print(df_svd.head())
    print(df_svd.shape)
    # sns.heatmap(df_svd)
    # plt.show()

    df_svd_scaled = (df_svd-df_svd.mean())/df_svd.std()
    # print(df_svd_scaled.head())

    # svd #####################################################################
    U, S, V = np.linalg.svd(df_svd_scaled, full_matrices=False)

    # analyzing explained variance ############################################
    var_explained = S/np.sum(S)
    var_explained = var_explained[:20]

    plt.figure()
    sns.barplot(x=list(range(1, len(var_explained)+1)),
                y=var_explained)
    plt.xlabel('SVs', fontsize=16)
    plt.ylabel('Percent Variance Explained', fontsize=16)

    # visualizing matrices ####################################################
    plt.figure()
    sns.heatmap(U, cmap="RdBu")
    plt.title("U")
    plt.yticks(
        list(map(lambda x: x+.5, range(len(new_idx)))),
        new_idx,
        rotation='horizontal')
    plt.savefig("heatmap_u.png")

    plt.figure()
    sns.heatmap(V[:, :20], cmap="RdBu")
    plt.title("V")

    # is there something? #####################################################
    plt.figure()
    df_svd_t = df_svd.transpose()
    sns.scatterplot(
        x='mean_degree',
        y='fills',
        hue='ecbs_cost',
        data=df_svd_t,
        palette="viridis",
        s=10)

    plt.figure()
    df_svd_t = df_svd.transpose()
    sns.scatterplot(
        x='mean_degree',
        y='diff_indep',
        hue='ecbs_cost',
        data=df_svd_t,
        palette="viridis",
        s=10)

    plt.figure()
    df_svd_t = df_svd.transpose()
    sns.scatterplot(
        x='n_agentss',
        y='diff_indep',
        hue='mean_degree',
        data=df_svd_t,
        palette="viridis",
        s=10)

    plt.figure()
    df_svd_t = df_svd.transpose()
    sns.scatterplot(
        x='mean_degree',
        y='diff_sim_decen_learned',
        hue='n_agentss',
        data=df_svd_t,
        palette="viridis",
        s=10)
    plt.show()
