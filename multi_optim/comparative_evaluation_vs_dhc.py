import logging
from math import floor
from random import Random
from typing import List, Optional

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
from sim.decentralized.policy import LearnedPolicy
from sim.decentralized.runner import run_a_scenario, to_agent_objects

from multi_optim.multi_optim_run import ITERATOR_TYPE, RADIUS

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


if __name__ == '__main__':
    # parameters
    logger.setLevel(logging.INFO)
    results_name: str = 'tiny'
    n_agents: int = 4
    n_eval: int = 10

    rng = Random(0)
    flann = FLANN(random_seed=0)

    # load graph
    graph_fname = f"multi_optim/results/{results_name}_graph.gpickle"
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
        torch.load(f"multi_optim/results/{results_name}_policy_model.pt"))

    # get map image
    yaml_fname = f"multi_optim/results/{results_name}_stats.yaml"
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
                             for i in range(g_dhc_by_nodes.number_of_nodes())}
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
    plt.imshow(
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
        node_color='r',)
    plt.savefig(f"multi_optim/results/{results_name}_dhcmap.png")

    lens_our = [None] * n_eval  # type: List[Optional[float]]
    lens_dhc_by_nodes = [None] * n_eval  # type: List[Optional[float]]
    lens_dhc_by_edge_len = [None] * n_eval  # type: List[Optional[float]]

    for i_e in range(n_eval):
        # sampling agents
        points: np.ndarray = sample_points(
            n_agents * 2, map_img, rng).detach().numpy()
        nn_our, _ = flann.nn(pos_our_np, points, 1)
        nn_dhc_by_nodes, _ = flann.nn(pos_dhc_by_nodes_np, points, 1)
        nn_dhc_by_edge_len, _ = flann.nn(pos_dhc_by_edge_len_np, points, 1)

        # eval ours
        starts_our = nn_our[:n_agents]
        goals_our = nn_our[n_agents:]
        agents = to_agent_objects(
            g_our,
            starts_our.tolist(),
            goals_our.tolist(),
            radius=RADIUS,
            rng=rng)
        assert agents is not None
        for agent in agents:
            agent.policy = LearnedPolicy(
                agent, policy_nn)
        paths_our: List[PATH] = []
        res_our = run_a_scenario(
            g_our, agents, False, ITERATOR_TYPE, paths_out=paths_our)
        logger.info(f"{res_our=}")
        logger.info(f"{paths_our=}")
        total_lenght_our = None  # type: Optional[float]
        if res_our[IDX_SUCCESS]:
            total_lenght_our = res_our[IDX_AVERAGE_LENGTH] * n_agents
            for i_a in range(n_agents):
                # before start
                total_lenght_our += float(np.linalg.norm(
                    np.array(points[i_a]) -
                    pos_our_np[starts_our[i_a]]))
                # after goal
                total_lenght_our += float(np.linalg.norm(
                    np.array(points[i_a + n_agents]) -
                    np.array(pos_our[goals_our[i_a]], dtype=np.float32)))

        # eval dhc by nodes
        starts_dhc_by_nodes = nn_dhc_by_nodes[:n_agents]
        goals_dhc_by_nodes = nn_dhc_by_nodes[n_agents:]
        res_dhc_by_nodes = dhc_eval(
            gridmap_by_nodes,
            np.array(
                [coords_from_node_by_nodes[n] for n in starts_dhc_by_nodes]),
            np.array(
                [coords_from_node_by_nodes[n] for n in goals_dhc_by_nodes]))
        logger.info(f"{res_dhc_by_nodes=}")
        total_lenght_dhc_by_nodes = None  # type: Optional[float]
        paths_dhc_by_nodes = None  # type: Optional[np.ndarray]
        if res_dhc_by_nodes != INVALID:
            _, _, paths_dhc_by_nodes = res_dhc_by_nodes
            total_lenght_dhc_by_nodes = get_total_len(
                n_agents, coords_from_node_by_nodes, points,
                pos_dhc_by_nodes_np, starts_dhc_by_nodes, goals_dhc_by_nodes,
                paths_dhc_by_nodes)

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
                g_our,
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
                coordpaths_dhc_by_edge_len.append(points[i_a + n_agents])

                # plot the path
                ax_our.plot(
                    [coordpaths_our[i][0] for i in range(len(coordpaths_our))],
                    [coordpaths_our[i][1] for i in range(len(coordpaths_our))],
                    c=colors[i_a],
                    linewidth=2)
                ax_dhc_by_nodes.plot(
                    [coordpaths_dhc_by_nodes[i][0]
                        for i in range(len(coordpaths_dhc_by_nodes))],
                    [coordpaths_dhc_by_nodes[i][1]
                        for i in range(len(coordpaths_dhc_by_nodes))],
                    c=colors[i_a],
                    linewidth=2)
                ax_dhc_by_edge_len.plot(
                    [coordpaths_dhc_by_edge_len[i][0]
                        for i in range(len(coordpaths_dhc_by_edge_len))],
                    [coordpaths_dhc_by_edge_len[i][1]
                        for i in range(len(coordpaths_dhc_by_edge_len))],
                    c=colors[i_a],
                    linewidth=2)

            f_our.savefig(
                f"multi_optim/results/{results_name}"
                + f"_paths_our_{i_e}.png")
            f_dhc_by_nodes.savefig(
                f"multi_optim/results/{results_name}"
                + f"_paths_dhc_by_nodes_{i_e}.png")
            f_dhc_by_edge_len.savefig(
                f"multi_optim/results/{results_name}"
                + f"_paths_dhc_by_edge_len_{i_e}.png")

        # print results
        logger.info("="*60)
        logger.info(f"{total_lenght_our=}")
        logger.info(f"{total_lenght_dhc_by_nodes=}")
        logger.info(f"{total_lenght_dhc_by_edge_len=}")

        lens_our[i_e] = (
            total_lenght_our
            if total_lenght_our is not None else 0.0)
        lens_dhc_by_nodes[i_e] = (
            total_lenght_dhc_by_nodes
            if total_lenght_dhc_by_nodes is not None else 0.0)
        lens_dhc_by_edge_len[i_e] = (
            total_lenght_dhc_by_edge_len
            if total_lenght_dhc_by_edge_len is not None else 0.0)

    # plot
    plt.clf()
    n_maps = 3
    width = 1./(n_maps+1)
    xs = np.arange(float(n_eval))
    plt.bar(xs, lens_our, width,
            color='r', alpha=.5, label='our')
    xs += width
    plt.bar(xs, lens_dhc_by_edge_len, width,
            color='g', alpha=.5, label='dhc_by_edge_len')
    xs += width
    plt.bar(xs, lens_dhc_by_nodes, width,
            color='b', alpha=.5, label='dhc_by_nodes')
    plt.legend()
    plt.savefig(f"multi_optim/results/{results_name}_lens.png")
