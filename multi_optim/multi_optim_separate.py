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

from multi_optim.configs import configs
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
        map_fname: str,
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
        map_fname,
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
        map_fname,
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
        map_fname=map_fname,
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
        map_fname=map_fname,
        seed=seed,
        # load_roadmap=f"{save_folder}/{prefix_sep_roadmap}_graph.gpickle",
        prefix=prefix_sep_policy,
        save_folder=save_folder,
        pool_in=pool)


def plot(save_folder: str, prefix_s: List[str]):
    titles = ["both", "only roadmap", "only policy"]
    metrics = ["path_length", "policy_regret", "overall_success"]
    width = .8

    n_nodes_per_prefix = {}
    n_agents_per_prefix = {}
    stats_per_prefix = {}
    for prefix in prefix_s:
        # we need three files:
        stats_per_prefix[prefix] = []
        stats_per_prefix[prefix].append(yaml.load(
            open(f"{save_folder}/{prefix}_stats.yaml", "r"),
            yaml.SafeLoader))
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
    fig, axs = plt.subplots(1, len(metrics), figsize=(8, 4.5))
    assert isinstance(axs, np.ndarray)
    for i_m, metric in enumerate(metrics):
        tick_labels = []
        for i_p, prefix in enumerate(prefixes_sorted_by_n_nodes):
            n_nodes = n_nodes_per_prefix[prefix]
            tick_labels.append(str(n_nodes))
            n_agents = n_agents_per_prefix[prefix]
            if metric == "path_length":
                key = "roadmap_test_length"
            elif metric == "policy_regret":
                key = f"policy_regret_{n_agents}"
            elif metric == "overall_success":
                key = f"general_success_{n_agents}"
            assert key is not None
            for i_s, stat in enumerate(stats_per_prefix[prefix]):
                if i_p == 0:
                    title = titles[i_s]
                else:
                    title = None
                axs[i_m].bar(
                    i_p + (i_s-1) * width / len(prefix_s),
                    stat[key]['x'][-1],
                    width=width / len(prefix_s),
                    label=title,
                    color=colors[i_s])
        pretty_name = metric.replace("_", " ").capitalize()
        axs[i_m].set_title(pretty_name)
        axs[i_m].set_xlabel("n_nodes")
        axs[i_m].set_xticks(range(len(prefix_s)))
        axs[i_m].set_xticklabels(tick_labels)
        axs[i_m].set_ylabel(pretty_name)

    axs[0].legend(loc='lower left')

    fig.tight_layout()
    plt.savefig(f"{save_folder}/all_sep.png")


if __name__ == "__main__":
    # Multiprocessing
    tmp.set_start_method('spawn')
    pool = tmp.Pool(processes=min(tmp.cpu_count(), 16))
    save_folder: str = "multi_optim/results"

    prefix_s = [
        # "debug",
        "tiny",
        "small",
        "medium",
        "large"
    ]

    # for prefix in prefix_s:
    #     run_optimization_sep(
    #         **configs[prefix],
    #         pool=pool,
    #         save_folder=save_folder)

    plot(save_folder, prefix_s)

    pool.close()
    pool.terminate()
