#!/usr/bin/env python3

import argparse
import logging
import pickle
from typing import Any, Dict, Optional, Union

import torch
from matplotlib import pyplot as plt
from scenarios.visualization import get_colors, plot_env, plot_schedule, plot_with_paths

PLOT_FOVS_STR = "plot_fovs"
PLOT_SCENARIO_STR = "plot_scenario"
PLOT_GRAPH_STR = "plot_graph"
N_TO_PLOT = 8


def plot_fovs(X, Y):
    rows = X.shape[2]
    columns = X.shape[3]
    middle = (X[:, :, 0, 0].shape[0] - 1) / 2
    end = X[:, :, 0, 0].shape[0] - 1

    fig = plt.figure()
    for row in range(rows):
        for col in range(columns):
            plt.subplot(rows, columns, columns * row + col + 1)
            plt.imshow(X[:, :, row, col], cmap="gray")
            # cross
            plt.plot([0, end], [middle, middle], "r")
            plt.plot([middle, middle], [0, end], "r")
    # label
    fig.suptitle(f"y={Y}")
    plt.show()


def plot_graph_wo_pos_data(
    ax,
    data_edge_index,
    pos: Union[Dict[Any, Any], torch.Tensor],
    data_x,
    highlight_nodes: Dict[int, str] = {},
):
    if isinstance(pos, torch.Tensor):
        data_pos = pos
    else:
        data_pos = torch.tensor([list(p) for _, p in pos.items()])
    plot_graph(ax, data_edge_index, data_pos, data_x, highlight_nodes)


def plot_graph(
    ax, data_edge_index, data_pos, data_x, highlight_nodes: Dict[int, str] = {}
):
    # edges
    n_edges = data_edge_index.shape[1]
    for i_e in range(n_edges):
        ax.plot(
            [
                data_pos[data_edge_index[0, i_e], 0],
                data_pos[data_edge_index[1, i_e], 0],
            ],  # x
            [
                data_pos[data_edge_index[0, i_e], 1],
                data_pos[data_edge_index[1, i_e], 1],
            ],  # y
            "k",
        )
    # nodes
    ax.scatter(
        data_pos[:, 0],
        data_pos[:, 1],
        marker="o",
        color="white",
        edgecolor="k",
        zorder=90,
    )
    # highlight nodes
    for n, color in highlight_nodes.items():
        rnd = torch.rand_like(data_x[n, 0]) * 0.01
        ax.scatter(
            data_pos[n, 0] + rnd, data_pos[n, 1], color=color, zorder=100, marker="x"
        )
    # node features
    n_node_features = data_x.shape[1]
    max_d = torch.max(data_x).item()
    min_d = torch.min(data_x).item()
    max_data = max(max_d, abs(min_d))
    n_nodes = data_x.shape[0]
    colors = get_colors(n_node_features)
    dp = 0.15 / (n_node_features + 2)
    for i_x in range(n_node_features):
        color = colors[i_x]
        for i_n in range(n_nodes):
            strength = float(data_x[i_n, i_x] / max_data)
            if strength < 0:
                color_middle = tuple([1 - c for c in color])
            else:
                color_middle = color
            abs_strength = abs(strength)
            color_middle_a = tuple(
                list(color_middle[:3])
                + [
                    abs_strength,
                ]
            )
            ax.scatter(
                data_pos[i_n, 0] + dp * (i_x + 1),
                data_pos[i_n, 1] + dp * (i_x + 1),
                color=color_middle_a,
                edgecolor=color,
                s=60,
            )
    for n in range(n_nodes):
        ax.text(
            data_pos[n, 0] - dp, data_pos[n, 1] + dp, f"{n}", color="k", fontsize=10
        )


if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", help="mode", choices=[PLOT_FOVS_STR, PLOT_SCENARIO_STR, PLOT_GRAPH_STR]
    )
    parser.add_argument("fname_read_pkl", type=argparse.FileType("rb"))
    args = parser.parse_args()

    with open(args.fname_read_pkl.name, "rb") as f:
        d = pickle.load(f)

    if args.mode == PLOT_FOVS_STR:
        for i in range(N_TO_PLOT):
            plot_fovs(*d[i])
    elif args.mode == PLOT_SCENARIO_STR:
        for i in range(N_TO_PLOT):
            env = d[i]["gridmap"]
            paths = d[i]["indepAgentPaths"]
            ax = plt.subplot()
            plot_env(ax, env)
            plot_with_paths(env, paths)
            plot_schedule(d[i])
            plt.show()
            print(d[i])
    elif args.mode == PLOT_GRAPH_STR:
        fig = plt.figure()
        for i in range(N_TO_PLOT):
            ax = fig.add_subplot(2, int(N_TO_PLOT / 2), i + 1)
            data_pos = d[i].pos
            data_x = d[i].x
            data_edge_index = d[i].edge_index
            plot_graph(ax, data_edge_index, data_pos, data_x)
            ax.set_title(f"y = {d[i].y}")
        plt.show()
