import logging
from mmap import mmap
from platform import node
from random import Random
from typing import List, Optional

import networkx as nx
import numpy as np
import torch
import yaml
from definitions import (FREE, IDX_AVERAGE_LENGTH, IDX_SUCCESS, INVALID, PATH,
                         POS)
from matplotlib import pyplot as plt
from planner.dhc.eval import eval as dhc_eval
from planner.policylearn.edge_policy import EdgePolicyModel
from pyflann import FLANN
from roadmaps.var_odrm_torch.var_odrm_torch import (check_edge, is_coord_free,
                                                    read_map, sample_points)
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


def make_gridmap_with_n_nodes_on_img(n_nodes, map_img, rng: Random):
    """
    Creates a gridmap with n_nodes on an map_img.
    """
    # starting with too few nodes
    n_rows_cols = np.floor(np.sqrt(n_nodes)).astype(int)
    n_nodes_free = 0
    max_tries_per_size = 100
    try_nr = 0
    pos_s = None
    while n_nodes_free < n_nodes:
        if try_nr > max_tries_per_size:
            n_rows_cols += 1
            try_nr = 0
        try_nr += 1
        res = 1.0 / (n_rows_cols + try_nr / max_tries_per_size)
        size = res * (n_rows_cols - 1)
        origin = np.array([rng.uniform(0, 1.0 - size),
                          rng.uniform(0, 1.0 - size)])
        pos_s = gridmap_from_origin_and_size(origin, size, n_rows_cols)
        n_nodes_free = 0
        for pos in pos_s:
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
        pos_s = gridmap_from_origin_and_size(origin, size, n_rows_cols)
        n_nodes_free = 0
        for pos in pos_s:
            if is_coord_free(map_img, pos):
                n_nodes_free += 1
        logger.debug(f"{origin=}")
        logger.debug(f"{res=}")
        logger.debug(f"{size=}")
        logger.debug(f"{n_rows_cols=}")
        logger.debug(f"{n_nodes_free=}")
        logger.debug(f"------------------")
    assert pos_s is not None
    pos_out = []
    nodes_out = []
    edgelist_out = []
    gridmap = np.ones((n_rows_cols, n_rows_cols))
    coords_from_node = []
    for n, pos in enumerate(pos_s):
        if is_coord_free(map_img, pos):
            pos_out.append(pos)
            nodes_out.append(n)
            x = nodes_out.index(n) % n_rows_cols
            y = nodes_out.index(n) // n_rows_cols
            gridmap[x, y] = FREE
            coords_from_node.append((x, y))
            # check edge left
            if n > 0:
                if check_edge(pos_s, map_img, n, n-1):
                    edgelist_out.append(
                        (nodes_out.index(n),
                         nodes_out.index(n-1))
                    )
            # check edge up
            if n >= n_rows_cols:
                if check_edge(pos_s, map_img, n, n-n_rows_cols):
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
    g = nx.read_gpickle(graph_fname)
    assert isinstance(g, nx.Graph)
    pos_graph = nx.get_node_attributes(g, POS)
    n_nodes = g.number_of_nodes()

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
    (pos_grid, edgelist, gridmap, coords_from_node
     ) = make_gridmap_with_n_nodes_on_img(
        n_nodes, map_img, rng)
    g_grid = nx.from_edgelist(edgelist)
    assert isinstance(g_grid, nx.Graph)
    pos_dict = {i: pos_grid[i, :] for i in range(g_grid.number_of_nodes())}
    nx.set_node_attributes(g_grid, pos_dict, POS)

    # plot
    plt.imshow(
        np.swapaxes(np.array(map_img), 0, 1),
        cmap='gray',
        origin='lower',
        alpha=.5,
        extent=(0, 1, 0, 1))
    nx.draw_networkx(
        g_grid,
        pos=pos_dict,
        with_labels=False,
        node_size=5,
        edge_color='r',
        node_color='r',)
    plt.savefig(f"multi_optim/results/{results_name}_gridmap.png")

    for i_e in range(n_eval):
        # sampling agents
        points: np.ndarray = sample_points(
            n_agents * 2, map_img, rng).detach().numpy()
        pos_graph_np = np.array([pos_graph[i] for i in range(n_nodes)])
        nn_graph, _ = flann.nn(pos_graph_np, points, 1)
        pos_grid_np = np.array([pos_grid[i]
                                for i in range(n_nodes)], dtype=np.float32)
        nn_grid, _ = flann.nn(pos_grid_np, points, 1)

        # eval ours
        starts = nn_graph[:n_agents]
        goals = nn_graph[n_agents:]
        agents = to_agent_objects(
            g,
            starts.tolist(),
            goals.tolist(),
            radius=RADIUS,
            rng=rng)
        assert agents is not None
        for agent in agents:
            agent.policy = LearnedPolicy(
                agent, policy_nn)
        paths: List[PATH] = []
        res_sim = run_a_scenario(
            g, agents, False, ITERATOR_TYPE, paths_out=paths)
        logger.info(f"{res_sim=}")
        logger.info(f"{paths=}")
        total_lenght_graph = None  # type: Optional[float]
        if res_sim[IDX_SUCCESS]:
            total_lenght_graph = res_sim[IDX_AVERAGE_LENGTH] * n_agents
            for i_a in range(n_agents):
                # before start
                total_lenght_graph += float(np.linalg.norm(
                    np.array(points[i_a]) -
                    pos_graph_np[starts[i_a]]))
                # after goal
                total_lenght_graph += float(np.linalg.norm(
                    np.array(points[i_a + n_agents]) -
                    np.array(pos_graph[goals[i_a]], dtype=np.float32)))

        # eval dhc
        starts = nn_grid[:n_agents]
        goals = nn_grid[n_agents:]
        res_dhc = dhc_eval(gridmap,
                           np.array(
                               [coords_from_node[n] for n in starts]),
                           np.array(
                               [coords_from_node[n] for n in goals]))
        logger.info(f"{res_dhc=}")
        total_lenght_grid = None  # type: Optional[float]
        if res_dhc != INVALID:
            _, _, paths_grid = res_dhc
            total_lenght_grid = 0.
            assert isinstance(paths_grid, np.ndarray)
            for i_a, path in enumerate(paths_grid):
                # before start
                total_lenght_grid += float(np.linalg.norm(
                    np.array(points[i_a]) -
                    pos_grid_np[starts[i_a]]))
                # path itself
                prev_pos = None  # type: Optional[np.ndarray]
                assert isinstance(path, np.ndarray)
                for pos in path.tolist():
                    coord_pos = pos_grid_np[
                        coords_from_node.index(tuple(pos))]
                    if prev_pos is not None:
                        total_lenght_grid += float(np.linalg.norm(
                            coord_pos -
                            prev_pos))
                    prev_pos = coord_pos
                # after goal
                total_lenght_grid += float(np.linalg.norm(
                    np.array(points[i_a + n_agents]) -
                    pos_grid_np[goals[i_a]]))

        # print results
        logger.info("="*80)
        logger.info(f"{total_lenght_graph=}")
        logger.info(f"{total_lenght_grid=}")
