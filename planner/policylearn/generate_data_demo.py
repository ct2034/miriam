#!/usr/bin/env python3

import argparse
import logging
import pickle

from matplotlib import pyplot as plt
from scenarios.visualization import (get_colors, plot_env, plot_schedule,
                                     plot_with_paths)

PLOT_FOVS_STR = "plot_fovs"
PLOT_SCENARIO_STR = "plot_scenario"
PLOT_GRAPH_STR = "plot_graph"
N_TO_PLOT = 8


def plot_fovs(X, Y):
    rows = X.shape[2]
    columns = X.shape[3]
    middle = (X[:, :, 0, 0].shape[0] - 1) / 2
    end = (X[:, :, 0, 0].shape[0] - 1)

    fig = plt.figure()
    for row in range(rows):
        for col in range(columns):
            plt.subplot(rows, columns, columns * row + col + 1)
            plt.imshow(X[:, :, row, col], cmap='gray')
            # cross
            plt.plot([0, end], [middle, middle], 'r')
            plt.plot([middle, middle], [0, end], 'r')
    # label
    fig.suptitle(f'y={Y}')
    plt.show()


def plot_graph(ax, data_edge_index, data_pos, data_x):
    # edges
    n_edges = data_edge_index.shape[1]
    for i_e in range(n_edges):
        ax.plot(
            [data_pos[data_edge_index[0, i_e], 0],
                data_pos[data_edge_index[1, i_e], 0]],  # x
            [data_pos[data_edge_index[0, i_e], 1],
                data_pos[data_edge_index[1, i_e], 1]],  # y
            'k'
        )
    # nodes
    ax.scatter(data_pos[:, 0], data_pos[:, 1], color='k')
    n_node_features = data_x.shape[1]
    n_nodes = data_x.shape[0]
    colors = get_colors(n_node_features)
    dp = 1./(n_node_features+2)
    for i_x in range(n_node_features):
        color = colors[i_x]
        ax.scatter(
            data_pos[:, 0] + dp*(i_x+1),
            data_pos[:, 1] + dp*(i_x+1),
            color=color,
            alpha=.1)
        for i_n in range(n_nodes):
            if data_x[i_n, i_x]:
                ax.scatter(
                    data_pos[i_n, 0] + dp*(i_x+1),
                    data_pos[i_n, 1] + dp*(i_x+1),
                    color=color)


if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', help='mode', choices=[
        PLOT_FOVS_STR, PLOT_SCENARIO_STR, PLOT_GRAPH_STR])
    parser.add_argument(
        'fname_read_pkl', type=argparse.FileType('rb'))
    args = parser.parse_args()

    with open(args.fname_read_pkl.name, 'rb') as f:
        d = pickle.load(f)

    if args.mode == PLOT_FOVS_STR:
        for i in range(N_TO_PLOT):
            plot_fovs(*d[i])
    elif args.mode == PLOT_SCENARIO_STR:
        for i in range(N_TO_PLOT):
            env = d[i]['gridmap']
            paths = d[i]['indepAgentPaths']
            ax = plt.subplot()
            plot_env(ax, env)
            plot_with_paths(env, paths)
            plot_schedule(d[i])
            plt.show()
            print(d[i])
    elif args.mode == PLOT_GRAPH_STR:
        fig = plt.figure()
        for i in range(N_TO_PLOT):
            ax = fig.add_subplot(2, int(N_TO_PLOT/2), i+1)
            data_pos = d[i].pos
            data_x = d[i].x
            data_edge_index = d[i].edge_index
            plot_graph(ax, data_edge_index, data_pos, data_x)
            ax.set_title(f"y = {d[i].y}")
        plt.show()
