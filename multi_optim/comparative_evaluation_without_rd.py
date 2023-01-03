import logging
import os
import pickle
from math import floor
from random import Random
from typing import List, Optional

import matplotlib
import networkx as nx
import numpy as np
import yaml
from matplotlib import pyplot as plt
from pyflann import FLANN

from definitions import POS
from roadmaps.var_odrm_torch.var_odrm_torch import read_map, sample_points

matplotlib.use('cairo')
plt.style.use('bmh')

logger = logging.getLogger(__name__)


def eval(results_name, base_folder, suffix_no_rd,
         figure_folder, n_eval):
    rng = Random(0)
    flann = FLANN(random_seed=0)

    # load roadmaps
    g_w_rd = nx.read_gpickle(f'{base_folder}/{results_name}_graph.gpickle')
    g_wo_rd = nx.read_gpickle(
        f'{base_folder}/{results_name}{suffix_no_rd}_graph.gpickle')
    pos_w_rd = nx.get_node_attributes(g_w_rd, POS)
    pos_wo_rd = nx.get_node_attributes(g_wo_rd, POS)
    pos_w_rd_np = np.array(list(pos_w_rd.values()))
    pos_wo_rd_np = np.array(list(pos_wo_rd.values()))

    # get map image
    yaml_fname = f"{base_folder}/{results_name}_stats.yaml"
    with open(yaml_fname, 'r') as f:
        stats = yaml.load(f, Loader=yaml.SafeLoader)
    map_fname = stats['static']['map_fname']
    map_img = None
    if map_fname.endswith(".png"):
        map_img = read_map(map_fname)
    assert map_img is not None

    lengths = []
    for i_e in range(n_eval):
        # get start and goal positions
        unique_starts_and_goals = False
        points: Optional[np.ndarray] = None
        nn_w_rd: Optional[np.ndarray] = None
        nn_wo_rd: Optional[np.ndarray] = None
        start_w_rd: Optional[np.ndarray] = None
        goal_w_rd: Optional[np.ndarray] = None
        start_wo_rd: Optional[np.ndarray] = None
        goal_wo_rd: Optional[np.ndarray] = None
        while not unique_starts_and_goals:
            points = sample_points(2, map_img, rng).detach().numpy()
            nn_w_rd, _ = flann.nn(pos_w_rd_np, points, 1)
            nn_wo_rd, _ = flann.nn(pos_wo_rd_np, points, 1)
            start_w_rd = nn_w_rd[0]
            goal_w_rd = nn_w_rd[1]
            start_wo_rd = nn_wo_rd[0]
            goal_wo_rd = nn_wo_rd[1]
            unique_starts_and_goals = start_w_rd != goal_w_rd and \
                start_wo_rd != goal_wo_rd

        # plan paths
        path_w_rd = nx.shortest_path(g_w_rd, start_w_rd, goal_w_rd)
        path_wo_rd = nx.shortest_path(g_wo_rd, start_wo_rd, goal_wo_rd)

        len_w_rd = (
            np.linalg.norm(points[0] - pos_w_rd[path_w_rd[0]]) +
            np.linalg.norm(points[1] - pos_w_rd[path_w_rd[-1]])
        )  # bits to and from graph
        for i in range(len(path_w_rd) - 1):
            len_w_rd += np.linalg.norm(pos_w_rd[path_w_rd[i]] -
                                       pos_w_rd[path_w_rd[i + 1]])
        len_wo_rd = (
            np.linalg.norm(points[0] - pos_wo_rd[path_wo_rd[0]]) +
            np.linalg.norm(points[1] - pos_wo_rd[path_wo_rd[-1]])
        )  # bits to and from graph
        for i in range(len(path_wo_rd) - 1):
            len_wo_rd += np.linalg.norm(pos_wo_rd[path_wo_rd[i]] -
                                        pos_wo_rd[path_wo_rd[i + 1]])
        lengths.append((len_w_rd, len_wo_rd))

        # store results
        with open(f'{figure_folder}/{results_name}_lengths.pkl', 'wb') as f:
            pickle.dump(lengths, f)


def plot(figure_folder, results_name):
    with open(f'{figure_folder}/{results_name}_lengths.pkl', 'rb') as f:
        lengths = pickle.load(f)

    fig, ax = plt.subplots()
    ax.set_title('Path length comparison')
    ax.set_xlabel('Path length with Grey Scott intialization')
    ax.set_ylabel('Path length with random initialization')
    ax.set_aspect('equal')
    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.5)
    ax.scatter([l[0] for l in lengths], [l[1] for l in lengths], s=1)
    # add trendline
    x = np.array([l[0] for l in lengths])
    y = np.array([l[1] for l in lengths])
    m, b = np.polyfit(x, y, 1)
    ax.plot(x, m * x + b, 'r--', linewidth=0.5)
    fig.savefig(f'{figure_folder}/{results_name}_lengths.pdf')


if __name__ == '__main__':
    logging.getLogger("sim.decentralized.runner").setLevel(logging.DEBUG)

    # parameters
    logger.setLevel(logging.INFO)
    results_name: str = 'debug'
    suffix_no_rd: str = '_no_rd'
    base_folder: str = 'multi_optim/results'
    figure_folder: str = f'{base_folder}/eval_without_rd'
    if not os.path.exists(figure_folder):
        os.makedirs(figure_folder)
    n_eval: int = 1000

    eval(results_name, base_folder, suffix_no_rd,
         figure_folder, n_eval)
    plot(figure_folder, results_name)
