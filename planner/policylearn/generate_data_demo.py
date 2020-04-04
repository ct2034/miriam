#!/usr/bin/env python3

import argparse
import logging
import pickle

from matplotlib import pyplot as plt

PLOT_FOVS_STR = "plot_fovs"


def plot_fovs(one_fov_data):
    rows = one_fov_data.shape[2]
    columns = one_fov_data.shape[3]
    middle = (one_fov_data[:, :, 0, 0].shape[0] - 1) / 2
    end = (one_fov_data[:, :, 0, 0].shape[0] - 1)

    for row in range(rows):
        for col in range(columns):
            plt.subplot(rows, columns, columns * row + col + 1)
            plt.imshow(one_fov_data[:, :, row, col], cmap='gray')
            # cross
            plt.plot([0, end], [middle, middle], 'r')
            plt.plot([middle, middle], [0, end], 'r')
    plt.show()


if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', help='mode', choices=[
        PLOT_FOVS_STR, ])
    parser.add_argument(
        'fname_read_pkl', type=argparse.FileType('rb'))
    args = parser.parse_args()

    with open(args.fname_read_pkl.name, 'rb') as f:
        d = pickle.load(f)

    if args.mode == PLOT_FOVS_STR:
        for i in range(4):
            plot_fovs(d[i][0])
