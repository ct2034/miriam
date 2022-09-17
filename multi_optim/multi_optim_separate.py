import logging
import sys
from itertools import product

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
        save_folder: str = "multi_optim/results"):

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

    plot(save_folder, prefix)


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


def plot(save_folder: str, prefix: str):
    # we need three files:
    stats = []
    stats.append(yaml.load(
        open(f"{save_folder}/{prefix}_stats.yaml", "r"),
        yaml.SafeLoader))
    stats.append(yaml.load(
        open(f"{save_folder}/{prefix}_sep_roadmap_stats.yaml", "r"),
        yaml.SafeLoader))
    stats.append(yaml.load(
        open(f"{save_folder}/{prefix}_sep_no_rm_policy_stats.yaml", "r"),
        yaml.SafeLoader))
    titles = ["both", "only roadmap", "only policy"]
    assert len(stats) == len(titles)
    n_agents = stats[0]["static"]["max_n_agents"]

    metrics = ["path_length", "policy_regret", "overall_success"]

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  # type: ignore
    fig, axs = plt.subplots(1, len(metrics), figsize=(8, 4.5))
    assert isinstance(axs, np.ndarray)
    for i, metric in enumerate(metrics):
        if metric == "path_length":
            key = "roadmap_test_length"
        elif metric == "policy_regret":
            key = f"policy_regret_{n_agents}"
        elif metric == "overall_success":
            key = f"general_success_{n_agents}"
        for j, stat in enumerate(stats):
            axs[i].plot(stat[key]["t"], stat[key]["x"],
                        label=titles[j], color=colors[j],
                        alpha=0.5)
        pretty_name = metric.replace("_", " ").capitalize()
        axs[i].set_title(pretty_name)
        axs[i].set_xlabel("iterations")
        axs[i].set_ylabel(pretty_name)
        axs[i].legend()

    fig.tight_layout()
    plt.savefig(f"{save_folder}/{prefix}_sep.png")


if __name__ == "__main__":
    # Multiprocessing
    tmp.set_start_method('spawn')
    pool = tmp.Pool(processes=min(tmp.cpu_count(), 16))

    for prefix in [
        "debug",
        "tiny",
        "small",
        "medium",
        "large"
    ]:
        run_optimization_sep(
            **configs[prefix],
            pool=pool)

    pool.close()
    pool.terminate()
