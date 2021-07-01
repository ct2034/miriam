#!/usr/bin/env python3

import argparse
import logging
import pickle

from matplotlib import pyplot as plt
from scenarios.visualization import plot_env, plot_schedule, plot_with_paths

PLOT_FOVS_STR = "plot_fovs"
PLOT_SCENARIO_STR = "plot_scenario"
N_TO_PLOT = 6


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


if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', help='mode', choices=[
        PLOT_FOVS_STR, PLOT_SCENARIO_STR])
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
