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
from planner.bvc.plan_bvc import plan as plan_bvc
from planner.policylearn.edge_policy import EdgePolicyModel
from pyflann import FLANN
from roadmaps.var_odrm_torch.var_odrm_torch import (check_edge, is_coord_free,
                                                    read_map, sample_points)
from scenarios.visualization import get_colors
from sim.decentralized.policy import LearnedPolicy, OptimalPolicy
from sim.decentralized.runner import run_a_scenario, to_agent_objects

from multi_optim.multi_optim_run import ITERATOR_TYPE

matplotlib.use('cairo')
plt.style.use('bmh')

logger = logging.getLogger(__name__)


def get_total_len(starts_coords, goals_coords, paths_bvc):
    total_lenght = 0.
    assert isinstance(paths_bvc, np.ndarray)
    for i_a, path in enumerate(paths_bvc):
        # before start
        total_lenght += float(np.linalg.norm(
            np.array(starts_coords[i_a]) -
            paths_bvc[i_a][0]))
        # path itself
        prev_pos = None  # type: Optional[np.ndarray]
        assert isinstance(path, np.ndarray)
        for pos in path.tolist():
            assert len(pos) == 2, "Position must be 2D."
            if prev_pos is not None:
                total_lenght += float(np.linalg.norm(
                    np.array(pos) -
                    np.array(prev_pos)))
            prev_pos = pos
            # after goal
        total_lenght += float(np.linalg.norm(
            np.array(goals_coords[i_a]) -
            paths_bvc[i_a][-1]))
    return total_lenght


def eval(logger, results_name, base_folder, figure_folder, n_agents_s, n_eval):
    rng = Random(0)
    flann = FLANN(random_seed=0)

    # load graph
    graph_fname = f"{base_folder}/{results_name}_graph.gpickle"
    g_our = nx.read_gpickle(graph_fname)
    assert isinstance(g_our, nx.Graph)
    n_nodes = g_our.number_of_nodes()
    pos_our = nx.get_node_attributes(g_our, POS)
    pos_our_np = np.array([pos_our[i] for i in range(n_nodes)])
    logger.info(f"{n_nodes=}")

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

    lens_our: np.ndarray = np.zeros((len(n_agents_s), n_eval))
    lens_bvc: np.ndarray = np.zeros((len(n_agents_s), n_eval))

    for i_na, n_agents in enumerate(n_agents_s):
        for i_e in range(n_eval):
            # sampling agents
            unique_starts_and_goals = False
            points: Optional[np.ndarray] = None
            nn_our: Optional[np.ndarray] = None
            starts_our: Optional[np.ndarray] = None
            goals_our: Optional[np.ndarray] = None
            starts_coords: Optional[np.ndarray] = None
            goals_coords: Optional[np.ndarray] = None
            while not unique_starts_and_goals:
                points = sample_points(
                    n_agents * 2, map_img, rng).detach().numpy()
                nn_our, _ = flann.nn(pos_our_np, points, 1)
                assert nn_our is not None
                assert points is not None
                starts_our = nn_our[:n_agents]
                goals_our = nn_our[n_agents:]
                starts_coords = points[:n_agents]
                goals_coords = points[n_agents:]
                unique_starts_and_goals = (
                    len(set(nn_our)) == 2 * n_agents)
            assert starts_our is not None
            assert goals_our is not None
            assert starts_coords is not None
            assert goals_coords is not None

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
                lens_our[i_na, i_e] = total_lenght_our

            # eval bvc
            res_bvc = plan_bvc(map_img, starts_coords.tolist(),
                               goals_coords.tolist(), radius=RADIUS)
            if isinstance(res_bvc, np.ndarray):  # not invalid
                lens_bvc[i_na][i_e] = get_total_len(
                    starts_coords, goals_coords, res_bvc)

    yaml.dump({
        'n_agents_s': n_agents_s,
        'lens_our': lens_our.tolist(),
        'lens_bvc': lens_bvc.tolist(),
    }, open(f"{figure_folder}/{results_name}_lens.yaml", 'w'))


def plot(figure_folder: str, results_name: str):
    yaml_data = yaml.load(
        open(f"{figure_folder}/{results_name}_lens.yaml"),
        Loader=yaml.SafeLoader)
    n_agents_s = yaml_data['n_agents_s']
    lens_our = np.array(yaml_data['lens_our'])
    lens_bvc = np.array(yaml_data['lens_bvc'])

    # plot
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  # type: ignore
    f, axs = plt.subplots(len(n_agents_s), 1, figsize=(9, len(n_agents_s)*6))
    assert isinstance(axs, np.ndarray)
    n_maps = 2
    width = 1./(n_maps+1)
    for i_na, n_agents in enumerate(n_agents_s):
        xs = np.arange(float(n_eval))
        axs[i_na].bar(xs, lens_our[i_na]/n_agents, width,
                      color=colors[0], alpha=1, label='ORDP')
        xs += width
        axs[i_na].bar(xs, lens_bvc[i_na]/n_agents, width,
                      color=colors[1], alpha=1, label='bvc')
        axs[i_na].set_xticks(np.arange(float(n_eval)))
        axs[i_na].set_xlabel('Trials')
        axs[i_na].set_ylabel('Average Pathlength')
        axs[i_na].set_title(f"{n_agents=}")
        axs[i_na].legend()
    f.savefig(f"{figure_folder}/{results_name}_lens.png")
    plt.close(f)

    # plot stats
    n_samples = np.shape(lens_our)[1]
    f, axs = plt.subplots(1, 2, figsize=(8, 4.5))
    assert isinstance(axs, np.ndarray)
    both_successfull = np.logical_and(lens_our != 0, lens_bvc != 0)
    # 0:our, 1:bvc
    data_success = [[0 for _ in range(len(n_agents_s))] for _ in range(2)]
    data_lenghts = [[list() for _ in range(len(n_agents_s))] for _ in range(2)]
    for i_na, n_agents in enumerate(n_agents_s):
        data_success[0][i_na] = np.count_nonzero(
            lens_our[i_na, :] != 0) / n_samples
        data_success[1][i_na] = np.count_nonzero(
            lens_bvc[i_na, :] != 0) / n_samples
        for i_e in range(n_samples):
            if both_successfull[i_na, i_e]:
                data_lenghts[0][i_na].append(lens_our[i_na, i_e])
                data_lenghts[1][i_na].append(lens_bvc[i_na, i_e])
    lenghts_min_max = [
        min(min([min(x) for x in data_lenghts[0] if len(x) > 0]),
            min([min(x) for x in data_lenghts[1] if len(x) > 0])),
        max(max([max(x) for x in data_lenghts[0] if len(x) > 0]),
            max([max(x) for x in data_lenghts[1] if len(x) > 0]))
    ]
    for i_na in range(len(n_agents_s)):
        for i_a in range(2):
            if(len(data_lenghts[i_a][i_na]) == 0):
                data_lenghts[i_a][i_na].append(99)

    width = .8
    axs[0].bar(np.array(n_agents_s)-width/2,
               data_success[0], width=width, label='ORDP')
    axs[0].bar(np.array(n_agents_s)+width/2,
               data_success[1], width=width, label='BVC')
    axs[0].set_xlim(1, max(n_agents_s)+1)
    axs[0].set_xticks(n_agents_s)
    axs[0].set_xlabel('Number of Agents')
    axs[0].set_ylabel('Success Rate')
    axs[0].legend(loc='center right')
    elemtns = axs[1].boxplot(data_lenghts[0],
                             positions=np.array(n_agents_s)-width/2,
                             widths=width)
    for el in elemtns['boxes']:
        el.set_color(colors[0])
    for el in elemtns['medians']:
        el.set_color(colors[0])
    for el in elemtns['whiskers']:
        el.set_color(colors[0])
    for el in elemtns['caps']:
        el.set_color(colors[0])
    for el in elemtns['fliers']:
        el.set_markeredgecolor(colors[0])
    elemtns = axs[1].boxplot(data_lenghts[1],
                                positions=np.array(n_agents_s)+width/2,
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
    axs[1].plot(0, 99, color=colors[0], label='ORDP')  # for legend
    axs[1].plot(0, 99, color=colors[1], label='BVC')  # for legend
    axs[1].set_xlim(1, max(n_agents_s)+1)
    axs[1].set_ylim(lenghts_min_max[0]-.1, lenghts_min_max[1]+.1)
    axs[1].set_xticks(n_agents_s)
    axs[1].set_xlabel('Number of Agents')
    axs[1].set_ylabel('Average Pathlength')
    axs[1].legend()
    f.tight_layout()
    f.savefig(f"{figure_folder}/{results_name}_lens_stats.pdf")
    plt.close(f)
    plt.close(f)


if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger("planner.bvc.plan_bvc").setLevel(logging.DEBUG)
    logging.getLogger("sim.decentralized.runner").setLevel(logging.DEBUG)

    # parameters
    logger.setLevel(logging.INFO)
    results_name: str = 'large'
    base_folder: str = 'multi_optim/results'
    figure_folder: str = f'{base_folder}/eval_vs_bvc'
    if not os.path.exists(figure_folder):
        os.makedirs(figure_folder)
    n_agents_s: List[int] = [2, 4, 6, 8]
    n_eval: int = 10

    # eval(logger, results_name, base_folder,
    #      figure_folder, n_agents_s, n_eval)
    plot(figure_folder, results_name)
