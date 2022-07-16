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

    # run_separately(
    #     n_nodes,
    #     n_runs_pose,
    #     n_runs_policy,
    #     n_episodes_per_run_policy,
    #     n_epochs_per_run_policy,
    #     batch_size_policy,
    #     stats_and_eval_every,
    #     lr_pos,
    #     lr_policy,
    #     max_n_agents,
    #     map_fname,
    #     seed,
    #     prefix,
    #     pool,
    #     save_folder)

    evaluate(map_fname, prefix, save_folder)
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
    prefix_sep_policy = f"{prefix}_sep_policy"
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
        load_roadmap=f"{save_folder}/{prefix_sep_roadmap}_graph.gpickle",
        prefix=prefix_sep_policy,
        save_folder=save_folder,
        pool_in=pool)

    pool.close()


def evaluate(map_fname, prefix, save_folder):
    prefix_sep_roadmap = f"{prefix}_sep_roadmap"
    prefix_sep_policy = f"{prefix}_sep_policy"
    # load final roadmap
    roadmap_sep = nx.read_gpickle(
        f"{save_folder}/{prefix_sep_roadmap}_graph.gpickle")
    assert isinstance(roadmap_sep, nx.Graph)
    flann_sep = FLANN()
    pos_sep = nx.get_node_attributes(roadmap_sep, POS)
    pos_sep_np = np.array([pos_sep[i]
                          for i in roadmap_sep.nodes()], dtype=np.float32)
    flann_sep.build_index(pos_sep_np, random_seed=0)
    roadmap_joint = nx.read_gpickle(
        f"{save_folder}/{prefix}_graph.gpickle")
    assert isinstance(roadmap_joint, nx.Graph)
    flann_joint = FLANN()
    pos_joint = nx.get_node_attributes(roadmap_joint, POS)
    pos_joint_np = np.array([pos_joint[i]
                            for i in roadmap_joint.nodes()], dtype=np.float32)
    flann_joint.build_index(pos_joint_np, random_seed=0)
    # load final policy
    gpu = torch.device("cpu")
    policy_sep = EdgePolicyModel(gpu=gpu)
    policy_sep.load_state_dict(
        torch.load(
            f"{save_folder}/{prefix_sep_policy}_policy_model.pt",
            map_location=gpu))
    policy_joint = EdgePolicyModel(gpu=gpu)
    policy_joint.load_state_dict(
        torch.load(
            f"{save_folder}/{prefix}_policy_model.pt",
            map_location=gpu))
    # load stats
    stats_static = yaml.load(
        open(f"{save_folder}/{prefix}_stats.yaml", "r"), yaml.SafeLoader)["static"]
    max_n_agents = stats_static["max_n_agents"]
    map_img = read_map(stats_static["map_fname"])

    # eval roadmap lenghts
    eval_sep = Eval(
        roadmap_sep,
        map_img,
        [8],
        1,
        ITERATOR_TYPE,
        RADIUS)
    eval_joint = Eval(
        roadmap_joint,
        map_img,
        [8],
        1,
        ITERATOR_TYPE,
        RADIUS)
    # first: own
    _ = eval_sep.evaluate_roadmap(
        roadmap_sep, flann_sep)
    _ = eval_joint.evaluate_roadmap(
        roadmap_joint, flann_joint)
    # second: the other
    roadmap_joint_over_sep = eval_sep.evaluate_roadmap(
        roadmap_joint, flann_joint)
    roadmap_sep_over_joint = eval_joint.evaluate_roadmap(
        roadmap_sep, flann_sep)

    # eval policy
    n_eval = 20
    n_agents_s = list(range(2, max_n_agents + 1, 2))
    res_sep_s = {}
    res_joint_s = {}
    for metric in ["regret", "success", "accuracy"]:
        res_sep_s[metric] = {}
        res_joint_s[metric] = {}
        for n_agents in n_agents_s:
            res_sep_s[metric][n_agents] = []
            res_joint_s[metric][n_agents] = []
    for i_e, n_agents in product(range(n_eval), n_agents_s):
        eval_sep = Eval(
            roadmap_sep,
            map_img,
            [n_agents],
            1,
            ITERATOR_TYPE,
            RADIUS,
            seed=i_e)
        eval_joint = Eval(
            roadmap_joint,
            map_img,
            [n_agents],
            1,
            ITERATOR_TYPE,
            RADIUS,
            seed=i_e)
        res_sep = eval_sep.evaluate_policy(
            policy_sep)
        res_joint = eval_joint.evaluate_policy(
            policy_joint)
        for metric in ["regret", "success", "accuracy"]:
            key = f"{metric}_{n_agents}"
            try:
                res_sep_s[metric][n_agents].append(res_sep[key])
            except KeyError:
                pass
            try:
                res_joint_s[metric][n_agents].append(res_joint[key])
            except KeyError:
                pass

    # save results
    with open(f"{save_folder}/{prefix}_joint_vs_sep.yaml", "w") as f:
        yaml.dump({
            "roadmap_joint_over_sep": roadmap_joint_over_sep,
            "roadmap_sep_over_joint": roadmap_sep_over_joint,
            "res_sep_s": res_sep_s,
            "res_joint_s": res_joint_s
        }, f)


def plot(save_folder: str, prefix: str):
    data = yaml.load(open(
        f"{save_folder}/{prefix}_joint_vs_sep.yaml", "r"),
        yaml.SafeLoader)
    roadmap_joint_over_sep = data["roadmap_joint_over_sep"]
    roadmap_sep_over_joint = data["roadmap_sep_over_joint"]
    res_sep_s = data["res_sep_s"]
    res_joint_s = data["res_joint_s"]

    stats_static = yaml.load(
        open(f"{save_folder}/{prefix}_stats.yaml", "r"), yaml.SafeLoader)["static"]
    max_n_agents = stats_static["max_n_agents"]
    n_agents_s = list(range(2, max_n_agents + 1, 2))

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  # type: ignore
    fig, axs = plt.subplots(1, 4, figsize=(40, 10))
    assert isinstance(axs, np.ndarray)
    axs[0].set_xlim(-0.5, 1.5)
    axs[0].plot(0, roadmap_joint_over_sep, "o", label="roadmap_joint_over_sep")
    axs[0].plot(1, roadmap_sep_over_joint, "o", label="roadmap_sep_over_joint")
    axs[0].legend()
    axs[0].set_title("Lengths")
    for i_m, metric in enumerate(["regret", "success", "accuracy"]):
        j_data_list = []
        s_data_list = []
        key_list = []
        n_data = len(n_agents_s) * 2
        for n_agents in n_agents_s:
            j_data_list.append(res_joint_s[metric][n_agents])
            key_list.append(f"j{n_agents}")
            s_data_list.append(res_sep_s[metric][n_agents])
            key_list.append(f"s{n_agents}")
        ticks = np.arange(n_data)
        j_positions = np.arange(0, n_data, 2)
        s_positions = np.arange(1, n_data, 2)
        j_parts = axs[i_m + 1].violinplot(
            j_data_list,
            positions=j_positions,
            showmeans=True)
        for pc in j_parts['bodies']:
            pc.set_facecolor(colors[0])
            pc.set_edgecolor(colors[0])
            pc.set_alpha(.3)
            pc.set_linewidths(0)
        j_parts['cbars'].set_color(colors[0])
        j_parts['cmeans'].set_color(colors[0])
        j_parts['cmins'].set_color(colors[0])
        j_parts['cmaxes'].set_color(colors[0])
        s_parts = axs[i_m + 1].violinplot(
            s_data_list,
            positions=s_positions,
            showmeans=True)
        for pc in s_parts['bodies']:
            pc.set_facecolor(colors[1])
            pc.set_edgecolor(colors[1])
            pc.set_alpha(.3)
            pc.set_linewidths(0)
        s_parts['cbars'].set_color(colors[1])
        s_parts['cmeans'].set_color(colors[1])
        s_parts['cmins'].set_color(colors[1])
        s_parts['cmaxes'].set_color(colors[1])
        axs[i_m + 1].set_title(metric.capitalize())
        axs[i_m + 1].set_xticks(ticks, key_list)
    fig.tight_layout()
    plt.savefig(f"{save_folder}/{prefix}_joint_vs_sep.png")


if __name__ == "__main__":
    # Multiprocessing
    tmp.set_start_method('spawn')
    pool = tmp.Pool(processes=min(tmp.cpu_count(), 16))

    for prefix in [
        # "debug",
        "tiny",
        # "small",
        # "medium",
        "large"
    ]:
        run_optimization_sep(
            **configs[prefix],
            pool=pool)

    pool.close()
    pool.terminate()
