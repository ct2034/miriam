import logging
from cProfile import label
from mmap import mmap
from platform import node
from random import Random
from typing import List, Optional

import networkx as nx
import numpy as np
import torch
import yaml
from definitions import (FREE, IDX_AVERAGE_LENGTH, IDX_SUCCESS, INVALID,
                         OBSTACLE, PATH, POS, N)
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


def gridmap_from_origin_and_size(origin, size, n_rows_cols):
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
    return np.array(pos_out), edgelist_out, gridmap, coords_from_node


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
    pos_our = nx.get_node_attributes(g_our, POS)
    n_nodes = g_our.number_of_nodes()

    # load policy
    policy_nn = EdgePolicyModel()
    policy_nn.load_state_dict(
        torch.load(f"multi_optim/results/{results_name}_policy_model.pt"))

    # get map image
    yaml_fname = f"multi_optim/results/{results_name}_stats.yaml"
    with open(yaml_fname, 'r') as f:
        stats = yaml.load(f, Loader=yaml.SafeLoader)
    map_img = read_map(stats['static']['map_fname'])

    # make equivalent gridmap
    logger.info(f"{n_nodes=}")
    (pos_dhc, edgelist, gridmap, coords_from_node
     ) = make_dhcmap_with_n_nodes_on_img(
        n_nodes, map_img, rng)
    g_dhc = nx.from_edgelist(edgelist)
    assert isinstance(g_dhc, nx.Graph)
    pos_dhc_dict = {i: pos_dhc[i, :] for i in range(g_dhc.number_of_nodes())}
    nx.set_node_attributes(g_dhc, pos_dhc_dict, POS)

    # plot
    plt.imshow(
        np.swapaxes(np.array(map_img), 0, 1),
        cmap='gray',
        origin='lower',
        alpha=.5,
        extent=(0, 1, 0, 1))
    nx.draw_networkx(
        g_dhc,
        pos=pos_dhc_dict,
        with_labels=False,
        node_size=5,
        edge_color='r',
        node_color='r',)
    plt.savefig(f"multi_optim/results/{results_name}_dhcmap.png")

    lens_our = [None] * n_eval  # type: List[Optional[float]]
    lens_dhc = [None] * n_eval  # type: List[Optional[float]]
    for i_e in range(n_eval):
        # sampling agents
        points: np.ndarray = sample_points(
            n_agents * 2, map_img, rng).detach().numpy()
        pos_our_np = np.array([pos_our[i] for i in range(n_nodes)])
        nn_our, _ = flann.nn(pos_our_np, points, 1)
        pos_dhc_np = np.array([pos_dhc[i]
                               for i in range(n_nodes)], dtype=np.float32)
        nn_dhc, _ = flann.nn(pos_dhc_np, points, 1)

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

        # eval dhc
        starts_dhc = nn_dhc[:n_agents]
        goals_dhc = nn_dhc[n_agents:]
        res_dhc = dhc_eval(gridmap,
                           np.array(
                               [coords_from_node[n] for n in starts_dhc]),
                           np.array(
                               [coords_from_node[n] for n in goals_dhc]))
        logger.info(f"{res_dhc=}")
        total_lenght_dhc = None  # type: Optional[float]
        paths_dhc = None  # type: Optional[np.ndarray]
        if res_dhc != INVALID:
            _, _, paths_dhc = res_dhc
            total_lenght_dhc = 0.
            assert isinstance(paths_dhc, np.ndarray)
            for i_a, path in enumerate(paths_dhc):
                # before start
                total_lenght_dhc += float(np.linalg.norm(
                    np.array(points[i_a]) -
                    pos_dhc_np[starts_dhc[i_a]]))
                # path itself
                prev_pos = None  # type: Optional[np.ndarray]
                assert isinstance(path, np.ndarray)
                for pos in path.tolist():
                    coord_pos = pos_dhc_np[
                        coords_from_node.index(tuple(pos))]
                    if prev_pos is not None:
                        total_lenght_dhc += float(np.linalg.norm(
                            coord_pos -
                            prev_pos))
                    prev_pos = coord_pos
                # after goal
                total_lenght_dhc += float(np.linalg.norm(
                    np.array(points[i_a + n_agents]) -
                    pos_dhc_np[goals_dhc[i_a]]))

        colors = get_colors(n_agents)
        if total_lenght_our is not None and total_lenght_dhc is not None:
            f_our, ax_our = plt.subplots()
            f_dhc, ax_dhc = plt.subplots()
            for ax in [ax_our, ax_dhc]:
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
                g_dhc,
                pos=pos_dhc_dict,
                with_labels=False,
                node_size=5,
                edge_color='k',
                node_color='k',
                ax=ax_dhc)

            for i_a in range(n_agents):
                coordpaths_our = []
                coordpaths_dhc = []
                # start
                coordpaths_our.append(points[i_a])
                coordpaths_dhc.append(points[i_a])
                # path
                for pos in paths_our[i_a]:
                    coordpaths_our.append(pos_our_np[pos])
                assert paths_dhc is not None
                for pos in paths_dhc[i_a]:
                    coordpaths_dhc.append(
                        pos_dhc_np[coords_from_node.index(tuple(pos))])
                # goal
                coordpaths_our.append(points[i_a + n_agents])
                coordpaths_dhc.append(points[i_a + n_agents])

                ax_our.plot(
                    [coordpaths_our[i][0] for i in range(len(coordpaths_our))],
                    [coordpaths_our[i][1] for i in range(len(coordpaths_our))],
                    c=colors[i_a],
                    linewidth=2)
                ax_dhc.plot(
                    [coordpaths_dhc[i][0] for i in range(len(coordpaths_dhc))],
                    [coordpaths_dhc[i][1] for i in range(len(coordpaths_dhc))],
                    c=colors[i_a],
                    linewidth=2)

            f_our.savefig(
                f"multi_optim/results/{results_name}_paths_our_{i_e}.png")
            f_dhc.savefig(
                f"multi_optim/results/{results_name}_paths_dhc_{i_e}.png")

        # print results
        logger.info("="*60)
        logger.info(f"{total_lenght_our=}")
        logger.info(f"{total_lenght_dhc=}")

        lens_our[i_e] = (
            total_lenght_our if total_lenght_our is not None else 0.0)
        lens_dhc[i_e] = (
            total_lenght_dhc if total_lenght_dhc is not None else 0.0)

    # plot
    plt.clf()
    plt.bar(range(n_eval), lens_our, color='b', alpha=.5, label='our')
    plt.bar(range(n_eval), lens_dhc, color='r', alpha=.5, label='dhc')
    plt.legend()
    plt.savefig(f"multi_optim/results/{results_name}_lens.png")
