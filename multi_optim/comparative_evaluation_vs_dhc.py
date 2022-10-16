import logging
import os
from math import floor
from random import Random
from typing import List, Optional

import matplotlib
import networkx as nx
import numpy as np
import torch
import yaml
from definitions import (FREE, IDX_AVERAGE_LENGTH, IDX_SUCCESS, INVALID,
                         MAP_IMG, OBSTACLE, PATH, POS, N)
from matplotlib import pyplot as plt
from planner.dhc.eval import eval as dhc_eval
from planner.policylearn.edge_policy import EdgePolicyModel
from pyflann import FLANN
from roadmaps.var_odrm_torch.var_odrm_torch import (check_edge, is_coord_free,
                                                    read_map, sample_points)
from scenarios.visualization import get_colors
from sim.decentralized.policy import LearnedPolicy, OptimalPolicy
from sim.decentralized.runner import run_a_scenario, to_agent_objects

from multi_optim.multi_optim_run import ITERATOR_TYPE, RADIUS

matplotlib.use('cairo')
plt.style.use('bmh')

logger = logging.getLogger(__name__)


def gridmap_from_origin_and_size(origin: np.ndarray,
                                 size: float, n_rows_cols: int):
    """
    Creates a gridmap with given origin for given size.
    """
    n_nodes = n_rows_cols ** 2
    pos_s = np.zeros((n_nodes, 2))
    pos_s[:, 0] = np.repeat(
        np.linspace(origin[0], origin[0] + size, n_rows_cols), n_rows_cols)
    pos_s[:, 1] = np.tile(
        np.linspace(origin[1], origin[1] + size, n_rows_cols), n_rows_cols)
    return pos_s


def pos_s_to_gridmap(map_img, n_rows_cols, pos_s_grid):
    """
    From a given list of potential node postions `pos_s_grid`, check if they
    are free on `map_img`. The positions are in a grid of edge lenght
    `n_rows_cols`. Then this returns free poses, edgelist and gridmap as well
    as a mapping between node numbers and positions in the grid
    (`coords_from_node`).
    """
    pos_out = []
    nodes_out = []
    edgelist_out = []
    gridmap = np.full((n_rows_cols, n_rows_cols),
                      OBSTACLE, dtype=np.int8)
    coords_from_node = []
    for n, pos in enumerate(pos_s_grid):
        if is_coord_free(map_img, pos):
            pos_out.append(pos)
            nodes_out.append(n)
            x = n % n_rows_cols
            y = n // n_rows_cols
            gridmap[x, y] = FREE
            coords_from_node.append((x, y))
            # TODO: Checking edges here makes no real sense, because edges are
            # not defined in the gridmap. I think instead we would have to
            # change the grid shape until all these edges are actually valid.
            # But lets test it with more nodes first ...

            # check edge left
            if n > 0:
                if check_edge(pos_s_grid, map_img, n, n-1):
                    edgelist_out.append(
                        (nodes_out.index(n),
                         nodes_out.index(n-1))
                    )
            # check edge up
            if n >= n_rows_cols:
                if check_edge(pos_s_grid, map_img, n, n-n_rows_cols):
                    edgelist_out.append(
                        (nodes_out.index(n),
                         nodes_out.index(n-n_rows_cols))
                    )
    return pos_out, edgelist_out, gridmap, coords_from_node


def make_dhcmap_with_n_nodes_on_img(n_nodes, map_img, rng: Random):
    """
    Creates a gridmap with n_nodes on an map_img.
    """
    # starting with too few nodes
    n_rows_cols = np.floor(np.sqrt(n_nodes)).astype(int)
    n_nodes_free = 0
    max_tries_per_size = 100
    try_nr = 0
    pos_s_grid = None
    while n_nodes_free < n_nodes:
        if try_nr > max_tries_per_size:
            n_rows_cols += 1
            try_nr = 0
        try_nr += 1
        res = 1.0 / (n_rows_cols + try_nr / max_tries_per_size)
        size = res * (n_rows_cols - 1)
        origin = np.array([rng.uniform(0, 1.0 - size),
                           rng.uniform(0, 1.0 - size)])
        pos_s_grid = gridmap_from_origin_and_size(origin, size, n_rows_cols)
        n_nodes_free = 0
        for pos in pos_s_grid:
            if is_coord_free(map_img, pos):
                n_nodes_free += 1
        logger.debug(f"{origin=}")
        logger.debug(f"{res=}")
        logger.debug(f"{size=}")
        logger.debug(f"{n_rows_cols=}")
        logger.debug(f"{n_nodes_free=}")
        logger.debug(f"------------------")
    logger.debug(">----<")
    while n_nodes_free != n_nodes:
        res = 1.0 / (n_rows_cols + try_nr / max_tries_per_size)
        size = res * (n_rows_cols - 1)
        origin = np.array([rng.uniform(0, 1.0 - size),
                           rng.uniform(0, 1.0 - size)])
        pos_s_grid = gridmap_from_origin_and_size(origin, size, n_rows_cols)
        n_nodes_free = 0
        for pos in pos_s_grid:
            if is_coord_free(map_img, pos):
                n_nodes_free += 1
        logger.debug(f"{origin=}")
        logger.debug(f"{res=}")
        logger.debug(f"{size=}")
        logger.debug(f"{n_rows_cols=}")
        logger.debug(f"{n_nodes_free=}")
        logger.debug(f"------------------")
    assert pos_s_grid is not None
    pos_out, edgelist_out, gridmap, coords_from_node = pos_s_to_gridmap(
        map_img, n_rows_cols, pos_s_grid)
    return np.array(pos_out), edgelist_out, gridmap, coords_from_node


def make_dhcmap_with_edge_len_on_img(edge_len: float, map_img: MAP_IMG):
    n_rows_cols = int(floor(1. / edge_len)) + 1
    size = (n_rows_cols - 1)*edge_len
    margin = (1. - size) / 2
    origin = np.array([margin, margin])
    pos_s_grid = gridmap_from_origin_and_size(
        origin, size, n_rows_cols)
    pos_out, edgelist_out, gridmap, coords_from_node = pos_s_to_gridmap(
        map_img, n_rows_cols, pos_s_grid)
    return np.array(pos_out), edgelist_out, gridmap, coords_from_node


def get_total_len(n_agents, coords_from_node_by_nodes, points, pos_dhc_np,
                  starts_dhc_by_nodes, goals_dhc_by_nodes, paths_dhc_by_nodes):
    total_lenght_dhc = 0.
    assert isinstance(paths_dhc_by_nodes, np.ndarray)
    for i_a, path in enumerate(paths_dhc_by_nodes):
        # before start
        total_lenght_dhc += float(np.linalg.norm(
            np.array(points[i_a]) -
            pos_dhc_np[starts_dhc_by_nodes[i_a]]))
        # path itself
        prev_pos = None  # type: Optional[np.ndarray]
        assert isinstance(path, np.ndarray)
        for pos in path.tolist():
            coord_pos = pos_dhc_np[
                coords_from_node_by_nodes.index(tuple(pos))]
            if prev_pos is not None:
                total_lenght_dhc += float(np.linalg.norm(
                    coord_pos -
                    prev_pos))
            prev_pos = coord_pos
            # after goal
        total_lenght_dhc += float(np.linalg.norm(
            np.array(points[i_a + n_agents]) -
            pos_dhc_np[goals_dhc_by_nodes[i_a]]))
    return total_lenght_dhc


def eval(logger, results_name, base_folder, figure_folder, n_agents_s, n_eval):
    rng = Random(0)
    flann = FLANN(random_seed=0)

    # load graph
    graph_fname = f"{base_folder}/{results_name}_graph.gpickle"
    g_our = nx.read_gpickle(graph_fname)
    assert isinstance(g_our, nx.Graph)
    n_nodes = g_our.number_of_nodes()
    logger.info(f"{n_nodes=}")
    pos_our = nx.get_node_attributes(g_our, POS)
    pos_our_np = np.array([pos_our[i] for i in range(n_nodes)])
    avg_edge_len_our = float(np.mean(np.linalg.norm(
        [pos_our_np[a] - pos_our_np[b] for (a, b) in g_our.edges],
        axis=1
    )))
    logger.info(f"{n_nodes=}")
    logger.info(f"{avg_edge_len_our=}")

    # load policy
    policy_nn = EdgePolicyModel()
    policy_nn.load_state_dict(
        torch.load(
            f"{base_folder}/{results_name}_policy_model.pt",
            map_location=torch.device("cpu")))

    # get map image
    yaml_fname = f"{base_folder}/{results_name}_stats.yaml"
    with open(yaml_fname, 'r') as f:
        stats = yaml.load(f, Loader=yaml.SafeLoader)
    map_img = read_map(stats['static']['map_fname'])

    # make gridmap with same n_nodes
    (pos_dhc_by_nodes, edgelist_dhc_by_nodes, gridmap_by_nodes,
     coords_from_node_by_nodes) = make_dhcmap_with_n_nodes_on_img(
        n_nodes, map_img, rng)
    g_dhc_by_nodes = nx.from_edgelist(edgelist_dhc_by_nodes)
    assert isinstance(g_dhc_by_nodes, nx.Graph)
    pos_dhc_by_nodes_dict = {i: pos_dhc_by_nodes[i, :]
                             for i in range(pos_dhc_by_nodes.shape[0])}
    nx.set_node_attributes(g_dhc_by_nodes, pos_dhc_by_nodes_dict, POS)
    pos_dhc_by_nodes_np = np.array(
        [pos_dhc_by_nodes[i] for i in range(n_nodes)],
        dtype=np.float32)

    # make gridmap with same edge length
    (pos_dhc_by_edge_len, edgelist_dhc_by_edge_len, gridmap_by_edge_len,
     coords_from_node_by_edge_len) = make_dhcmap_with_edge_len_on_img(
         avg_edge_len_our, map_img)
    g_dhc_by_edge_len = nx.from_edgelist(edgelist_dhc_by_edge_len)
    assert isinstance(g_dhc_by_edge_len, nx.Graph)
    n_nodes_dhc_by_edge_len = g_dhc_by_edge_len.number_of_nodes()
    logger.info(f"{n_nodes_dhc_by_edge_len=}")
    pos_dhc_by_edge_len_dict = {i: pos_dhc_by_edge_len[i, :]
                                for i in range(n_nodes_dhc_by_edge_len)}
    nx.set_node_attributes(g_dhc_by_edge_len, pos_dhc_by_edge_len_dict, POS)
    pos_dhc_by_edge_len_np = np.array(
        [pos_dhc_by_edge_len[i]
            for i in range(n_nodes_dhc_by_edge_len)],
        dtype=np.float32)

    # plot
    f, ax = plt.subplots(1, 1)
    ax.imshow(
        np.swapaxes(np.array(map_img), 0, 1),
        cmap='gray',
        origin='lower',
        alpha=.5,
        extent=(0, 1, 0, 1))
    nx.draw_networkx(
        g_dhc_by_nodes,
        pos=pos_dhc_by_nodes_dict,
        with_labels=False,
        node_size=5,
        edge_color='r',
        node_color='r',
        ax=ax)
    f.savefig(f"{figure_folder}/{results_name}_dhcmap_by_nodes.png")
    plt.close(f)

    f, ax = plt.subplots(1, 1)
    ax.imshow(
        np.swapaxes(np.array(map_img), 0, 1),
        cmap='gray',
        origin='lower',
        alpha=.5,
        extent=(0, 1, 0, 1))
    nx.draw_networkx(
        g_dhc_by_edge_len,
        pos=pos_dhc_by_edge_len_dict,
        with_labels=False,
        node_size=5,
        edge_color='r',
        node_color='r',
        ax=ax)
    f.savefig(
        f"{figure_folder}/{results_name}_dhcmap_by_edge_len.png")
    plt.close(f)

    lens_our: np.ndarray = np.zeros((len(n_agents_s), n_eval))
    lens_dhc_by_nodes: np.ndarray = np.zeros((len(n_agents_s), n_eval))
    lens_dhc_by_edge_len: np.ndarray = np.zeros((len(n_agents_s), n_eval))

    for i_na, n_agents in enumerate(n_agents_s):
        for i_e in range(n_eval):
            # sampling agents
            unique_starts_and_goals = False
            points: Optional[np.ndarray] = None
            nn_our: Optional[np.ndarray] = None
            nn_dhc_by_nodes: Optional[np.ndarray] = None
            nn_dhc_by_edge_len: Optional[np.ndarray] = None
            starts_our: Optional[np.ndarray] = None
            goals_our: Optional[np.ndarray] = None
            while not unique_starts_and_goals:
                points = sample_points(
                    n_agents * 2, map_img, rng).detach().numpy()
                nn_our, _ = flann.nn(pos_our_np, points, 1)
                nn_dhc_by_nodes, _ = flann.nn(pos_dhc_by_nodes_np, points, 1)
                nn_dhc_by_edge_len, _ = flann.nn(
                    pos_dhc_by_edge_len_np, points, 1)
                starts_our = nn_our[:n_agents]
                goals_our = nn_our[n_agents:]
                unique_starts_and_goals = (
                    len(set(nn_our)) == 2 * n_agents)
            assert points is not None
            assert nn_our is not None
            assert nn_dhc_by_nodes is not None
            assert nn_dhc_by_edge_len is not None
            assert starts_our is not None
            assert goals_our is not None

            # eval ours
            agents = to_agent_objects(
                g_our,
                starts_our.tolist(),
                goals_our.tolist(),
                radius=RADIUS/100,
                rng=rng)
            total_lenght_our = None  # type: Optional[float]
            paths_our: Optional[List[PATH]] = None
            if agents is not None:
                for agent in agents:
                    agent.policy = LearnedPolicy(
                        agent, policy_nn)
                    # agent.policy = OptimalPolicy(agent, None)
                paths_our = []
                res_our = run_a_scenario(
                    g_our, agents,
                    plot=False,
                    iterator=ITERATOR_TYPE,
                    paths_out=paths_our)
                logger.info(f"{res_our=}")
                logger.info(f"{paths_our=}")
                if res_our[IDX_SUCCESS]:
                    total_lenght_our = res_our[IDX_AVERAGE_LENGTH] * n_agents
                    assert total_lenght_our is not None
                    for i_a in range(n_agents):
                        # before start
                        total_lenght_our += float(np.linalg.norm(
                            np.array(points[i_a]) -
                            pos_our_np[starts_our[i_a]]))  # type: ignore
                        # after goal
                        total_lenght_our += float(np.linalg.norm(
                            np.array(points[i_a + n_agents]) -
                            np.array(pos_our[goals_our[i_a]],
                                     dtype=np.float32)))  # type: ignore

            # eval dhc by nodes
            starts_dhc_by_nodes = nn_dhc_by_nodes[:n_agents]
            goals_dhc_by_nodes = nn_dhc_by_nodes[n_agents:]
            res_dhc_by_nodes = dhc_eval(
                gridmap_by_nodes,
                np.array(
                    [coords_from_node_by_nodes[n]
                     for n in starts_dhc_by_nodes]),
                np.array(
                    [coords_from_node_by_nodes[n]
                     for n in goals_dhc_by_nodes]))
            logger.info(f"{res_dhc_by_nodes=}")
            total_lenght_dhc_by_nodes = None  # type: Optional[float]
            paths_dhc_by_nodes = None  # type: Optional[np.ndarray]
            if res_dhc_by_nodes != INVALID:
                _, _, paths_dhc_by_nodes = res_dhc_by_nodes
                total_lenght_dhc_by_nodes = get_total_len(
                    n_agents, coords_from_node_by_nodes, points,
                    pos_dhc_by_nodes_np, starts_dhc_by_nodes,
                    goals_dhc_by_nodes, paths_dhc_by_nodes)

            # eval dhc by edge len
            starts_dhc_by_edge_len = nn_dhc_by_edge_len[:n_agents]
            goals_dhc_by_edge_len = nn_dhc_by_edge_len[n_agents:]
            res_dhc_by_edge_len = dhc_eval(
                gridmap_by_edge_len,
                np.array(
                    [coords_from_node_by_edge_len[n]
                     for n in starts_dhc_by_edge_len]),
                np.array(
                    [coords_from_node_by_edge_len[n]
                     for n in goals_dhc_by_edge_len]))
            logger.info(f"{res_dhc_by_edge_len=}")
            total_lenght_dhc_by_edge_len = None  # type: Optional[float]
            paths_dhc_by_edge_len = None  # type: Optional[np.ndarray]
            if res_dhc_by_edge_len != INVALID:
                _, _, paths_dhc_by_edge_len = res_dhc_by_edge_len
                total_lenght_dhc_by_edge_len = get_total_len(
                    n_agents, coords_from_node_by_edge_len, points,
                    pos_dhc_by_edge_len_np, starts_dhc_by_edge_len,
                    goals_dhc_by_edge_len, paths_dhc_by_edge_len)

            if i_e == 0:
                colors = get_colors(n_agents)
                if (total_lenght_our is not None and
                    total_lenght_dhc_by_nodes is not None and
                        total_lenght_dhc_by_edge_len is not None):
                    f_our, ax_our = plt.subplots()
                    f_dhc_by_nodes, ax_dhc_by_nodes = plt.subplots()
                    f_dhc_by_edge_len, ax_dhc_by_edge_len = plt.subplots()
                    for ax in [ax_our, ax_dhc_by_nodes, ax_dhc_by_edge_len]:
                        ax.imshow(
                            np.swapaxes(np.array(map_img), 0, 1),
                            cmap='gray',
                            origin='lower',
                            alpha=.5,
                            extent=(0, 1, 0, 1))
                    nx.draw_networkx(
                        nx.subgraph_view(
                            g_our, filter_edge=lambda a, b: a != b),
                        pos=nx.get_node_attributes(g_our, POS),
                        with_labels=False,
                        node_size=5,
                        edge_color='k',
                        node_color='k',
                        ax=ax_our)
                    nx.draw_networkx(
                        g_dhc_by_nodes,
                        pos=pos_dhc_by_nodes_dict,
                        with_labels=False,
                        node_size=5,
                        edge_color='k',
                        node_color='k',
                        ax=ax_dhc_by_nodes)
                    nx.draw_networkx(
                        g_dhc_by_edge_len,
                        pos=pos_dhc_by_edge_len_dict,
                        with_labels=False,
                        node_size=5,
                        edge_color='k',
                        node_color='k',
                        ax=ax_dhc_by_edge_len)

                    for i_a in range(n_agents):
                        coordpaths_our = []
                        coordpaths_dhc_by_nodes = []
                        coordpaths_dhc_by_edge_len = []
                        # start
                        coordpaths_our.append(points[i_a])
                        coordpaths_dhc_by_nodes.append(points[i_a])
                        coordpaths_dhc_by_edge_len.append(points[i_a])
                        # path
                        assert paths_our is not None
                        for node in paths_our[i_a]:
                            coordpaths_our.append(pos_our_np[node])
                        assert paths_dhc_by_nodes is not None
                        for pos in paths_dhc_by_nodes[i_a]:
                            coordpaths_dhc_by_nodes.append(
                                pos_dhc_by_nodes_np[
                                    coords_from_node_by_nodes.index(
                                        tuple(pos))])
                        assert paths_dhc_by_edge_len is not None
                        for pos in paths_dhc_by_edge_len[i_a]:
                            coordpaths_dhc_by_edge_len.append(
                                pos_dhc_by_edge_len_np[
                                    coords_from_node_by_edge_len.index(
                                        tuple(pos))])
                        # goal
                        coordpaths_our.append(points[i_a + n_agents])
                        coordpaths_dhc_by_nodes.append(points[i_a + n_agents])
                        coordpaths_dhc_by_edge_len.append(
                            points[i_a + n_agents])

                        # plot the path
                        ax_our.plot(
                            [coordpaths_our[i][0]
                                for i in range(len(
                                    coordpaths_our))],
                            [coordpaths_our[i][1]
                                for i in range(len(
                                    coordpaths_our))],
                            c=colors[i_a],
                            linewidth=2)
                        ax_dhc_by_nodes.plot(
                            [coordpaths_dhc_by_nodes[i][0]
                                for i in range(len(
                                    coordpaths_dhc_by_nodes))],
                            [coordpaths_dhc_by_nodes[i][1]
                                for i in range(len(
                                    coordpaths_dhc_by_nodes))],
                            c=colors[i_a],
                            linewidth=2)
                        ax_dhc_by_edge_len.plot(
                            [coordpaths_dhc_by_edge_len[i][0]
                                for i in range(len(
                                    coordpaths_dhc_by_edge_len))],
                            [coordpaths_dhc_by_edge_len[i][1]
                                for i in range(len(
                                    coordpaths_dhc_by_edge_len))],
                            c=colors[i_a],
                            linewidth=2)

                    f_our.savefig(
                        f"{figure_folder}/{results_name}"
                        + f"_paths_our_{n_agents=}.png")
                    f_dhc_by_nodes.savefig(
                        f"{figure_folder}/{results_name}"
                        + f"_paths_dhc_by_nodes_{n_agents=}.png")
                    f_dhc_by_edge_len.savefig(
                        f"{figure_folder}/{results_name}"
                        + f"_paths_dhc_by_edge_len_{n_agents=}.png")
                    plt.close(f_our)
                    plt.close(f_dhc_by_nodes)
                    plt.close(f_dhc_by_edge_len)

            # print results
            logger.info("="*60)
            logger.info(f"{total_lenght_our=}")
            logger.info(f"{total_lenght_dhc_by_nodes=}")
            logger.info(f"{total_lenght_dhc_by_edge_len=}")

            lens_our[i_na][i_e] = total_lenght_our
            lens_dhc_by_nodes[i_na][i_e] = total_lenght_dhc_by_nodes
            lens_dhc_by_edge_len[i_na][i_e] = total_lenght_dhc_by_edge_len

    yaml.dump({
        'n_agents_s': n_agents_s,
        'lens_our': lens_our.tolist(),
        'lens_dhc_by_edge_len': lens_dhc_by_edge_len.tolist(),
        'lens_dhc_by_nodes': lens_dhc_by_nodes.tolist()
    }, open(f"{figure_folder}/{results_name}_lens.yaml", 'w'))


def plot(figure_folder: str, results_name: str):
    yaml_data = yaml.load(
        open(f"{figure_folder}/{results_name}_lens.yaml"),
        Loader=yaml.SafeLoader)
    n_agents_s = yaml_data['n_agents_s']
    lens_our = np.array(yaml_data['lens_our'], dtype=np.float64)
    lens_dhc_by_edge_len = np.array(
        yaml_data['lens_dhc_by_edge_len'], dtype=np.float64)
    lens_dhc_by_nodes = np.array(
        yaml_data['lens_dhc_by_nodes'], dtype=np.float64)

    # plot
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  # type: ignore
    f, axs = plt.subplots(len(n_agents_s), 1, figsize=(9, len(n_agents_s)*6))
    assert isinstance(axs, np.ndarray)
    n_maps = 3
    width = 1./(n_maps+1)
    for i_na, n_agents in enumerate(n_agents_s):
        xs = np.arange(float(n_eval))
        axs[i_na].bar(xs, lens_our[i_na]/n_agents, width,
                      color=colors[0], alpha=1, label='our')
        xs += width
        axs[i_na].bar(xs, lens_dhc_by_edge_len[i_na]/n_agents, width,
                      color=colors[1], alpha=1, label='dhc_by_edge_len')
        xs += width
        axs[i_na].bar(xs, lens_dhc_by_nodes[i_na]/n_agents, width,
                      color=colors[2], alpha=1, label='dhc_by_nodes')
        axs[i_na].set_xticks(np.arange(float(n_eval)))
        axs[i_na].set_xlabel('Trials')
        axs[i_na].set_ylabel('Average Pathlength')
        axs[i_na].set_title(f"{n_agents=}")
        axs[i_na].legend()
    f.savefig(f"{figure_folder}/{results_name}_lens.png")
    plt.close(f)

    # plot stats
    n_samples = np.shape(lens_our)[1]
    f, axs = plt.subplots(1, 1, figsize=(4.5, 4.5))
    # assert isinstance(axs, np.ndarray)
    axs = [axs]
    success_our = np.logical_not(np.isnan(lens_our))
    success_dhc_by_edge_len = np.logical_not(np.isnan(lens_dhc_by_edge_len))
    success_dhc_by_nodes = np.logical_not(np.isnan(lens_dhc_by_nodes))
    diff_vs_dhc_by_edge_len = lens_dhc_by_edge_len - lens_our
    diff_vs_dhc_by_nodes = lens_dhc_by_nodes - lens_our

    # for i_na, n_agents in enumerate(n_agents_s):
    #     data_success[0][i_na] = np.count_nonzero(
    #         lens_our[i_na, :] != 0) / n_samples
    #     data_success[1][i_na] = np.count_nonzero(
    #         lens_bvc[i_na, :] != 0) / n_samples
    #     for i_e in range(n_samples):
    #         if both_successfull[i_na, i_e]:
    #             data_lenghts[0][i_na].append(lens_our[i_na, i_e])
    #             data_lenghts[1][i_na].append(lens_bvc[i_na, i_e])
    diffs_min_max = [
        min(min([min(x) for x in diff_vs_dhc_by_edge_len if len(x) > 0]),
            min([min(x) for x in diff_vs_dhc_by_nodes if len(x) > 0])),
        max(max([max(x) for x in diff_vs_dhc_by_edge_len if len(x) > 0]),
            max([max(x) for x in diff_vs_dhc_by_nodes if len(x) > 0]))
    ]
    diff_vs_dhc_by_edge_len_list = [list() for _ in range(len(n_agents_s))]
    diff_vs_dhc_by_nodes_list = [list() for _ in range(len(n_agents_s))]
    for i_na, n_agents in enumerate(n_agents_s):
        for i_e in range(n_samples):
            if success_our[i_na, i_e] and success_dhc_by_edge_len[i_na, i_e]:
                diff_vs_dhc_by_edge_len_list[i_na].append(
                    diff_vs_dhc_by_edge_len[i_na, i_e])
            if success_our[i_na, i_e] and success_dhc_by_nodes[i_na, i_e]:
                diff_vs_dhc_by_nodes_list[i_na].append(
                    diff_vs_dhc_by_nodes[i_na, i_e])

    width = .8
    # lengths
    axs[0].boxplot([0, 1],
                   positions=[-20],
                   widths=width)
    elemtns = axs[0].boxplot(diff_vs_dhc_by_edge_len_list,
                             positions=np.array(n_agents_s)-width/2,
                             widths=width)
    for el in elemtns['boxes']:
        el.set_color(colors[1])
    for el in elemtns['medians']:
        el.set_color(colors[1])
    for el in elemtns['whiskers']:
        el.set_color(colors[1])
    for el in elemtns['caps']:
        el.set_color(colors[1])
    for el in elemtns['fliers']:
        el.set_markeredgecolor(colors[1])

    elemtns = axs[0].boxplot(diff_vs_dhc_by_nodes_list,
                             positions=np.array(n_agents_s)+width/2,
                             widths=width)
    for el in elemtns['boxes']:
        el.set_color(colors[2])
    for el in elemtns['medians']:
        el.set_color(colors[2])
    for el in elemtns['whiskers']:
        el.set_color(colors[2])
    for el in elemtns['caps']:
        el.set_color(colors[2])
    for el in elemtns['fliers']:
        el.set_markeredgecolor(colors[2])

    axs[0].plot(0, 99, color=colors[1],
                label='DHC (gridmap by edge length)')  # for legend
    axs[0].plot(0, 99, color=colors[2],
                label='DHC (gridmap by nr of vertices)')  # for legend
    axs[0].set_xlim(1, max(n_agents_s)+1)
    axs[0].set_ylim(diffs_min_max[0]-.1, diffs_min_max[1]+.1)
    axs[0].set_xticks(n_agents_s)
    axs[0].set_xticklabels(n_agents_s)
    axs[0].set_xlabel('Number of Agents')
    axs[0].set_ylabel('Average Pathlength Difference DHC - ORDP')
    axs[0].legend()
    # success
    # width = 2. / 3.5
    # n_agents_s = np.array(n_agents_s)
    # axs[1].bar(
    #     n_agents_s-width,
    #     [np.count_nonzero(x)/n_samples for x in success_our],
    #     width=width, color=colors[0], label='our')
    # axs[1].bar(
    #     n_agents_s,
    #     [np.count_nonzero(x)/n_samples for x in success_dhc_by_edge_len],
    #     width=width, color=colors[1],
    #     label='DHC (by edge length)')
    # axs[1].bar(
    #     n_agents_s+width,
    #     [np.count_nonzero(x)/n_samples for x in success_dhc_by_nodes],
    #     width=width, color=colors[2],
    #     label='DHC (by nr of vertices)')
    # axs[1].set_xlim(1, max(n_agents_s)+1)
    # axs[1].set_ylim(0., 1.04)
    # axs[1].set_xticks(n_agents_s)
    # axs[1].set_xlabel('Number of Agents')
    # axs[1].set_ylabel('Success Rate')
    # axs[1].legend(loc='lower right')
    # done
    f.tight_layout()
    f.savefig(f"{figure_folder}/{results_name}_lens_stats.pdf")
    plt.close(f)


if __name__ == '__main__':
    logging.getLogger("sim.decentralized.runner").setLevel(logging.DEBUG)

    # parameters
    logger.setLevel(logging.INFO)
    results_name: str = 'large'
    base_folder: str = 'multi_optim/results'
    figure_folder: str = f'{base_folder}/eval_vs_dhc'
    if not os.path.exists(figure_folder):
        os.makedirs(figure_folder)
    n_agents_s: List[int] = [2, 4, 6, 8]
    n_eval: int = 10

    # eval(logger, results_name, base_folder,
    # figure_folder, n_agents_s, n_eval)
    plot(figure_folder, results_name)
