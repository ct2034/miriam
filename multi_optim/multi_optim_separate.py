import logging
import sys
from itertools import product
from typing import List

import matplotlib
import networkx as nx
import numpy as np
import torch
import torch.multiprocessing as tmp
import yaml
from definitions import POS
from matplotlib import pyplot as plt
from planner.policylearn.edge_policy import EdgePolicyModel
from pyflann import FLANN
from roadmaps.var_odrm_torch.var_odrm_torch import read_map

from multi_optim.configs import configs_all_maps
from multi_optim.eval import Eval
from multi_optim.multi_optim_run import RADIUS, run_optimization
from multi_optim.state import ITERATOR_TYPE

matplotlib.use('cairo')
plt.style.use('bmh')


def run_optimization_sep(
        n_nodes: int,
        n_runs_pose: int,
        n_runs_policy: int,
        n_episodes_per_run_policy: int,  # how many episodes to sample per run
        n_epochs_per_run_policy: int,  # how often to learn same data
        batch_size_policy: int,
        stats_and_eval_every: int,
        lr_pos: float,
        lr_policy: float,
        max_n_agents: int,
        map_name: str,
        seed: int,
        prefix: str,
        pool,
        save_folder: str):

    run_separately(
        n_nodes,
        n_runs_pose,
        n_runs_policy,
        n_episodes_per_run_policy,
        n_epochs_per_run_policy,
        batch_size_policy,
        stats_and_eval_every,
        lr_pos,
        lr_policy,
        max_n_agents,
        map_name,
        seed,
        prefix,
        pool,
        save_folder)


def run_separately(
        n_nodes,
        n_runs_pose,
        n_runs_policy,
        n_episodes_per_run_policy,
        n_epochs_per_run_policy,
        batch_size_policy,
        stats_and_eval_every,
        lr_pos,
        lr_policy,
        max_n_agents,
        map_name,
        seed,
        prefix,
        pool,
        save_folder):

    # 1. roadmap
    prefix_sep_roadmap = f"{prefix}_sep_roadmap"
    run_optimization(
        n_nodes=n_nodes,
        n_runs_pose=n_runs_pose,
        n_runs_policy=0,
        n_episodes_per_run_policy=n_episodes_per_run_policy,
        n_epochs_per_run_policy=1,
        batch_size_policy=batch_size_policy,
        stats_and_eval_every=stats_and_eval_every,
        lr_pos=lr_pos,
        lr_policy=lr_policy,
        max_n_agents=max_n_agents,
        map_name=map_name,
        seed=seed,
        prefix=prefix_sep_roadmap,
        save_folder=save_folder,
        pool_in=pool)

    # 2. policy
    prefix_sep_policy = f"{prefix}_sep_no_rm_policy"  # not use old roadmap
    run_optimization(
        n_nodes=n_nodes,
        n_runs_pose=0,
        n_runs_policy=n_runs_policy,
        n_episodes_per_run_policy=n_episodes_per_run_policy,
        n_epochs_per_run_policy=n_epochs_per_run_policy,
        batch_size_policy=batch_size_policy,
        stats_and_eval_every=stats_and_eval_every,
        lr_pos=lr_pos,
        lr_policy=lr_policy,
        max_n_agents=max_n_agents,
        map_name=map_name,
        seed=seed,
        # load_roadmap=f"{save_folder}/{prefix_sep_roadmap}_graph.gpickle",
        prefix=prefix_sep_policy,
        save_folder=save_folder,
        pool_in=pool)


def plot(save_folder: str, prefix_s: List[str]):
    titles = ["only roadmap", "only policy"]
    metrics = ["path_length", "general_success", "general_regret"]
    width = .8

    n_nodes_per_prefix = {}
    n_agents_per_prefix = {}
    stats_per_prefix = {}
    both_stats_per_prefix = {}
    for prefix in prefix_s:
        # we need three files:
        both_stats_per_prefix[prefix] = yaml.load(
            open(f"{save_folder}/{prefix}_stats.yaml", "r"),
            yaml.SafeLoader)
        stats_per_prefix[prefix] = []
        stats_per_prefix[prefix].append(yaml.load(
            open(f"{save_folder}/{prefix}_sep_roadmap_stats.yaml", "r"),
            yaml.SafeLoader))
        stats_per_prefix[prefix].append(yaml.load(
            open(f"{save_folder}/{prefix}_sep_no_rm_policy_stats.yaml", "r"),
            yaml.SafeLoader))

        n_nodes_per_prefix[prefix] = stats_per_prefix[prefix][0]['static']['n_nodes']

        assert len(stats_per_prefix[prefix]) == len(titles)

        n_agents: int = sys.maxsize
        for s in stats_per_prefix[prefix]:
            available_n_agents_policy_regret = list(filter(
                lambda x: x.startswith("policy_regret_"), s.keys()))
            available_n_agents_general_success = list(filter(
                lambda x: x.startswith("general_success_"), s.keys()))
            available_n_agents_policy_regret = map(int, map(
                lambda x: x.split("_")[-1], available_n_agents_policy_regret))
            available_n_agents_general_success = map(int, map(
                lambda x: x.split("_")[-1], available_n_agents_general_success))
            n_agents = min(
                n_agents,
                max(available_n_agents_policy_regret),
                max(available_n_agents_general_success))
        n_agents_per_prefix[prefix] = n_agents

    prefixes_sorted_by_n_nodes = sorted(
        prefix_s, key=lambda x: n_nodes_per_prefix[x])

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  # type: ignore
    f_width = 4.5 * len(metrics)
    fig, axs = plt.subplots(1, len(metrics), figsize=(f_width, 4.5))
    axs = [axs] if len(metrics) == 1 else axs
    # assert isinstance(axs, np.ndarray)
    for i_m, metric in enumerate(metrics):
        tick_labels = []
        for i_p, prefix in enumerate(prefixes_sorted_by_n_nodes):
            n_nodes = n_nodes_per_prefix[prefix]
            tick_labels.append(f"{n_nodes} ({n_agents_per_prefix[prefix]})")
            n_agents = n_agents_per_prefix[prefix]
            both_stats = both_stats_per_prefix[prefix]
            # less agents here for less noise in data
            # less_n_agents = n_agents - 2 if n_agents > 2 else n_agents
            if metric == "path_length":
                key = f"general_length_{n_agents}"
            elif metric == "policy_regret":
                key = f"policy_regret_{n_agents}"
            elif metric == "general_regret":
                key = f"general_regret_{n_agents}"
            elif metric == "overall_success":
                key = f"general_success_{n_agents}"
            elif metric == "policy_success":
                key = f"policy_success_{n_agents}"
            assert key is not None
            for i_s, stat in enumerate(stats_per_prefix[prefix]):
                stat_end = len(stat[key]['x'])
                stat_start = int(stat_end * 0.8)
                both_stats_end = len(both_stats[key]['x'])
                both_stats_start = both_stats_end - stat_end + stat_start
                data = (
                    np.array(stat[key]['x'])[stat_start:stat_end-1] -
                    np.array(both_stats[key]['x'])[
                        both_stats_start:both_stats_end-1]
                )  # type: ignore
                elements = axs[i_m].boxplot(
                    [data],
                    positions=[i_p + (i_s-.5) * width / len(titles)],
                    widths=[width / len(titles)],
                    labels=[stat])
                for el in elements['boxes']:
                    el.set_color(colors[i_s+2])
                for el in elements['medians']:
                    el.set_color(colors[i_s+2])
                for el in elements['whiskers']:
                    el.set_color(colors[i_s+2])
                for el in elements['caps']:
                    el.set_color(colors[i_s+2])
                for el in elements['fliers']:
                    el.set_markeredgecolor(colors[i_s+2])
        pretty_name = " ".join(
            map(lambda x: x.capitalize(), metric.split("_")))
        axs[i_m].plot(-5, 0, color=colors[2],
                      label='roadmap only - both')  # for legend
        axs[i_m].plot(-5, 0, color=colors[3],
                      label='policy only - both')  # for legend
        axs[i_m].set_xlim(-.5, len(n_agents_per_prefix)-.5)
        # axs[i_m].set_title("Path Length Surplus")
        axs[i_m].set_xlabel("Number of Vertices (Agents)")
        axs[i_m].set_xticks(range(len(prefix_s)))
        axs[i_m].set_xticklabels(tick_labels)
        axs[i_m].set_ylabel(pretty_name + " Difference Ablation - Baseline")

    axs[0].legend(loc='upper right')

    fig.tight_layout()
    plt.savefig(f"{save_folder}/ablation.pdf")


if __name__ == "__main__":
    # Multiprocessing
    tmp.set_start_method('spawn')
    pool = tmp.Pool(processes=min(tmp.cpu_count(), 16))
    save_folder: str = "multi_optim/results"

    prefix_s = [
        "debug_c",
        "tiny_c",
        "small_c",
        "medium_c",
        "large_c"
    ]

    for prefix in prefix_s:
        run_optimization_sep(
            **configs_all_maps[prefix],
            pool=pool,
            save_folder=save_folder)

    plot(save_folder, prefix_s)

    pool.close()
    pool.terminate()
