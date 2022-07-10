import logging
import sys

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

from multi_optim.eval import Eval
from multi_optim.multi_optim_run import RADIUS, run_optimization
from multi_optim.state import ITERATOR_TYPE

matplotlib.use('cairo')


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

    # 3. evaluate
    prefix_sep_eval = f"{prefix}_sep_eval"
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
    # evaluate
    eval_sep = Eval(
        roadmap=roadmap_sep,
        map_img=read_map(map_fname),
        n_agents_s=[2, 4],
        n_eval_per_n_agents=10,
        iterator_type=ITERATOR_TYPE,
        radius=RADIUS)
    eval_joint = Eval(
        roadmap=roadmap_joint,
        map_img=read_map(map_fname),
        n_agents_s=[2, 4],
        n_eval_per_n_agents=10,
        iterator_type=ITERATOR_TYPE,
        radius=RADIUS)
    res_sep = eval_sep.evaluate_both(
        policy_sep, roadmap_sep, flann_sep)
    res_joint = eval_joint.evaluate_both(
        policy_joint, roadmap_joint, flann_joint)
    # plot
    assert set(res_sep.keys()) == set(res_joint.keys())
    keys = sorted(res_sep.keys())
    f, axs = plt.subplots(1, len(keys), figsize=(5 * len(keys), 5))
    assert isinstance(axs, np.ndarray)
    for i_k, key in enumerate(keys):
        axs[i_k].bar(
            0, res_sep[key], color="C0", label="separate")
        axs[i_k].bar(
            1, res_joint[key], color="C1", label="joint")
        axs[i_k].set_title(key)
        axs[i_k].legend()
    f.tight_layout()
    f.savefig(f"{save_folder}/{prefix}_joint_vs_sep.png")
    plt.close(f)

    # 4. save results
    with open(f"{save_folder}/{prefix}_joint_vs_sep.yaml", "w") as f:
        yaml.dump({
            "separate": res_sep,
            "joint": res_joint,
        }, f)


if __name__ == "__main__":
    # Multiprocessing
    tmp.set_start_method('spawn')
    pool = tmp.Pool(processes=min(tmp.cpu_count(), 16))

    debug = False
    if debug:
        # debug run
        prefix = "debug"
        logging.getLogger(__name__).setLevel(logging.DEBUG)
        logging.getLogger("multi_optim.multi_optim_run").setLevel(logging.INFO)
        logging.getLogger(
            "planner.mapf_implementations.plan_cbs_roadmap"
        ).setLevel(logging.INFO)
        run_optimization_sep(
            n_nodes=8,
            n_runs_pose=8,
            n_runs_policy=8,
            n_episodes_per_run_policy=2,
            n_epochs_per_run_policy=2,
            batch_size_policy=16,
            stats_and_eval_every=4,
            lr_pos=1e-2,
            lr_policy=1e-3,
            max_n_agents=2,
            map_fname="roadmaps/odrm/odrm_eval/maps/x.png",
            seed=0,
            prefix=prefix,
            pool=pool)

    # large run
    prefix = "large"
    run_optimization_sep(
        n_nodes=128,
        n_runs_pose=64,
        n_runs_policy=64,
        n_episodes_per_run_policy=256,
        n_epochs_per_run_policy=4,
        batch_size_policy=128,
        stats_and_eval_every=2,
        lr_pos=1e-3,
        lr_policy=1e-4,
        max_n_agents=10,
        map_fname="roadmaps/odrm/odrm_eval/maps/x.png",
        seed=0,
        prefix=prefix,
        pool=pool)

    pool.close()
    pool.terminate()
