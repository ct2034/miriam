import copy
import datetime
import logging
import os
import pickle
import socket
import time
from random import Random
from typing import Dict, List, Optional, Tuple

import git.repo
import networkx as nx
import numpy as np
import scenarios
import scenarios.solvers
import tools
import torch
import torch.multiprocessing as tmp
from cuda_util import pick_gpu_lowest_memory
from definitions import INVALID, MAP_IMG, PATH_W_COORDS, POS
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from planner.policylearn.edge_policy import EdgePolicyDataset, EdgePolicyModel
from planner.policylearn.edge_policy_graph_utils import TIMEOUT
from pyflann import FLANN
from roadmaps.var_odrm_torch.var_odrm_torch import (draw_graph,
                                                    make_graph_and_flann,
                                                    optimize_poses_from_paths,
                                                    read_map, sample_points)
from sim.decentralized.iterators import IteratorType
from tools import ProgressBar, StatCollector
from torch_geometric.loader import DataLoader

if __name__ == "__main__":
    from eval import Eval
    from state import ACTION, ScenarioState
else:
    from multi_optim.eval import Eval
    from multi_optim.state import ACTION, ScenarioState

logger = logging.getLogger(__name__)

MAX_STEPS = 10
RADIUS = 0.001
ITERATOR_TYPE = IteratorType.LOOKAHEAD2


def sample_trajectory_proxy(args):
    return sample_trajectory(*args)


def sample_trajectory(seed, graph, n_agents, model, map_img: MAP_IMG,
                      max_steps=MAX_STEPS):
    """Sample a trajectory using the given policy."""
    rng = Random(seed)
    starts = None
    goals = None
    flann = FLANN()
    pos = graph.nodes.data(POS)
    pos_np = np.array([pos[n] for n in graph.nodes])
    flann.build_index(np.array(pos_np, dtype=np.float32),
                      random_index=0)

    starts_coord: Optional[List[Tuple[float, float]]] = None
    goals_coord: Optional[List[Tuple[float, float]]] = None

    solvable = False
    while not solvable:
        unique = False
        while not unique:
            starts_goals_coord = sample_points(n_agents * 2, map_img, rng)
            result, _ = flann.nn_index(
                starts_goals_coord.detach().numpy(),
                1,
                random_seed=0)
            starts_coord = [
                starts_goals_coord[i, :2].detach().numpy().astype(float)
                for i in range(n_agents)
            ]
            goals_coord = [
                starts_goals_coord[i, :2].detach().numpy().astype(float)
                for i in range(n_agents, n_agents*2)
            ]
            starts = result[0:n_agents].tolist()
            goals = result[n_agents:].tolist()
            unique = (len(set(starts)) == n_agents and
                      len(set(goals)) == n_agents)
        # is this solvable?
        optimal_paths = scenarios.solvers.cached_cbsr(
            graph, starts, goals, radius=RADIUS,
            timeout=int(TIMEOUT*.9))
        if optimal_paths != INVALID:
            solvable = True

    assert starts_coord is not None
    assert goals_coord is not None

    state = ScenarioState(graph, starts, goals, model, RADIUS)
    state.run()

    # Sample states
    these_ds = []
    paths = None  # type: Optional[List[PATH_W_COORDS]]
    for i_s in range(max_steps):
        try:
            if state.finished:
                if len(state.paths_out[0]) > 0:
                    paths = [(
                        starts_coord[i],
                        goals_coord[i],
                        state.paths_out[i])
                        for i in range(n_agents)]
                else:
                    paths = None
                break
            observations = state.observe()
            actions: Dict[int, ACTION] = {}
            assert observations is not None, "observations is None"
            for i_a, (d, bfs) in observations.items():
                # find actions to take using the policy
                actions[i_a] = model.predict(d.x, d.edge_index, bfs)
                # observation, action pairs for learning
                these_ds.append(d)
            state.step(actions)
        except RuntimeError as e:
            logger.warning("RuntimeError: {}".format(e))
            break
    return these_ds, paths


def _get_data_folder(save_folder, prefix):
    return f"{save_folder}/{prefix}_data"


def _get_path_data(save_folder, prefix, hash) -> str:
    folder = _get_data_folder(save_folder, prefix)
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder+f"/{hash}.pkl"


def sample_trajectories_in_parallel(
        model: EdgePolicyModel, graph: nx.Graph, map_img: MAP_IMG, flann,
        n_agents: int, n_episodes: int, prefix: str, require_paths: bool,
        save_folder, pool, rng: Random
) -> Tuple[str, List[List[PATH_W_COORDS]]]:
    model_copy = EdgePolicyModel()
    model_copy.load_state_dict(copy.deepcopy(model.state_dict()))
    model_copy.eval()

    params = [(s, graph, n_agents, model_copy, map_img)
              for s in rng.sample(
        range(2**32), k=n_episodes)]
    generation_hash = tools.hasher([], {
        "seeds": [p[0] for p in params],
        "graph": graph,
        "n_agents": n_agents,
        "model": model_copy
    })
    new_fname: str = _get_path_data(save_folder, prefix, generation_hash)
    # only create file if this data does not exist or if paths are required
    paths_s: List[List[PATH_W_COORDS]] = []
    if os.path.exists(new_fname) and not require_paths:
        pass
    else:
        results_s = pool.imap_unordered(
            sample_trajectory_proxy, params)
        # results_s = map(sample_trajectory_proxy, params)
        new_ds = []
        for ds, paths in results_s:
            new_ds.extend(ds)
            if paths is not None:
                paths_s.append(paths)
        with open(new_fname, "wb") as f:
            pickle.dump(new_ds, f)

    # add this to the dataset
    return new_fname, paths_s


def optimize_policy(model, batch_size, optimizer, epds
                    ) -> Tuple[EdgePolicyModel, float]:
    if len(epds) == 0:
        return model, 0.0
    loader = DataLoader(epds, batch_size=batch_size, shuffle=True)
    loss_s = []
    for _, batch in enumerate(loader):
        loss = model.learn(batch, optimizer)
        loss_s.append(loss)

    if len(loss_s) == 0:
        loss_s = [0]
    return model, np.mean(loss_s)


def run_optimization(
        n_nodes: int,
        n_runs_pose: int,
        n_runs_policy: int,
        n_epochs_per_run_policy: int,
        batch_size_policy: int,
        stats_and_eval_every: int,
        lr_pos: float,
        lr_policy: float,
        n_agents: int,
        map_fname: str,
        seed: int,
        load_policy_model: Optional[str] = None,
        prefix: str = "noname",
        save_images: bool = True,
        save_folder: Optional[str] = None):
    rng = Random(seed)
    logger.info(f"run_optimization {prefix}")
    torch.manual_seed(rng.randint(0, 2 ** 32))
    if save_folder is None:
        save_folder = "multi_optim/results"  # default

    # multiprocessing
    n_processes = min(tmp.cpu_count(), 8)
    pool = tmp.Pool(processes=n_processes)

    # Roadmap
    map_img: MAP_IMG = read_map(map_fname)
    pos = sample_points(n_nodes, map_img, rng)
    optimizer_pos = torch.optim.Adam([pos], lr=lr_pos)
    g: nx.Graph
    flann: FLANN
    (g, flann) = make_graph_and_flann(pos, map_img)

    # GPU or CPU?
    if torch.cuda.is_available():
        gpu = torch.device(pick_gpu_lowest_memory())
        logger.info(f"Using GPU {gpu}")
        torch.cuda.empty_cache()
    else:
        logger.warning("GPU not available, using CPU")
        gpu = torch.device("cpu")

    # Policy
    policy_model = EdgePolicyModel(gpu=gpu)
    if load_policy_model is not None:
        policy_model.load_state_dict(torch.load(load_policy_model))
    policy_model.to(gpu)
    for param in policy_model.parameters():
        param.share_memory_()
    policy_model.share_memory()

    # Optimizer
    optimizer_policy = torch.optim.Adam(
        policy_model.parameters(), lr=lr_policy)
    policy_loss: Optional[float] = None

    # Eval
    # little less agents for evaluation
    eval_n_agents = int(np.ceil(n_agents * .7))
    eval = Eval(g, map_img,
                n_agents=eval_n_agents, n_eval=10,
                iterator_type=ITERATOR_TYPE, radius=RADIUS)

    # Data for policy
    epds = EdgePolicyDataset(f"{save_folder}/{prefix}_data")

    # Visualization and analysis
    stats = StatCollector([
        "general_eval_time_perc",
        "general_length",
        "general_new_data_percentage",
        "general_regret",
        "general_success",
        "n_policy_data_len",
        "policy_accuracy",
        "policy_loss",
        "policy_regret",
        "policy_success",
        "roadmap_test_length",
        "roadmap_training_length"])
    stats.add_statics({
        # metadata
        "hostname": socket.gethostname(),
        "git_hash": git.repo.Repo(".").head.object.hexsha,
        "started_at": datetime.datetime.now().isoformat(),
        # parameters
        "n_nodes": n_nodes,
        "n_runs_pose": n_runs_pose,
        "n_runs_policy": n_runs_policy,
        "batch_size_policy": batch_size_policy,
        "stats_every": stats_and_eval_every,
        "lr_pos": lr_pos,
        "lr_policy": lr_policy,
        "n_agents": n_agents,
        "map_fname": map_fname,
        "load_policy_model": (
            load_policy_model if load_policy_model else "None"),
        "prefix": prefix
    })
    if save_images:
        draw_graph(g, map_img, title="Start")
        plt.savefig(f"{save_folder}/{prefix}_start.png")

    # Making sense of two n_runs
    n_runs = max(n_runs_pose, n_runs_policy)
    if n_runs_policy > n_runs_pose:
        n_runs_per_run_policy = 1
        n_runs_per_run_pose = n_runs // n_runs_pose
    else:  # n_runs_pose > n_runs_policy
        n_runs_per_run_pose = 1
        n_runs_per_run_policy = n_runs // n_runs_policy

    # Run optimization
    pb = ProgressBar(f"{prefix} Optimization", n_runs, 1)
    # roadmap_test_length = 0
    roadmap_training_length = 0
    for i_r in range(n_runs):
        start_time = time.process_time()
        optimize_poses_now: bool = i_r % n_runs_per_run_pose == 0
        optimize_policy_now: bool = i_r % n_runs_per_run_policy == 0

        # Sample runs for both optimizations
        # assert n_runs_policy >= n_runs_pose, \
        #     "otherwise we dont need optiomal solution that often"
        old_data_len = len(epds)
        new_fname, paths_s = sample_trajectories_in_parallel(
            policy_model, g, map_img, flann, n_agents, n_epochs_per_run_policy,
            prefix, optimize_poses_now, save_folder,  pool, rng)
        epds.add_file(new_fname)
        data_len = len(epds)
        if data_len > 0:
            new_data_percentage = (data_len - old_data_len) / data_len
        else:
            new_data_percentage = 0.

        # Optimizing Poses
        if optimize_poses_now:
            (g, pos, flann, roadmap_training_length
             ) = optimize_poses_from_paths(
                g, pos, paths_s, map_img, optimizer_pos)
            if i_r % stats_and_eval_every == 0:
                stats.add("roadmap_training_length", i_r,
                          float(roadmap_training_length))

        # Optimizing Policy
        if optimize_policy_now:
            policy_model, policy_loss = optimize_policy(
                policy_model, batch_size_policy, optimizer_policy, epds)

        if i_r % stats_and_eval_every == 0:
            end_optimization_time = time.process_time()

            if optimize_policy_now:
                # also eval now
                (policy_regret, policy_success, policy_accuracy
                 ) = eval.evaluate_policy(policy_model)
                assert policy_loss is not None
                stats.add("policy_loss", i_r, float(policy_loss))
                if policy_regret is not None:
                    stats.add("policy_regret", i_r, float(policy_regret))
                stats.add("policy_success", i_r, float(policy_success))
                stats.add("policy_accuracy", i_r, policy_accuracy)
                logger.info(f"(P) Loss: {policy_loss:.3f}")
                logger.info(f"(P) Regret: {policy_regret:e}")
                logger.info(f"(P) Success: {policy_success}")
                logger.info(f"(P) Accuracy: {policy_accuracy:.3f}")

            if optimize_poses_now:
                # eval the current roadmap
                roadmap_test_length = eval.evaluate_roadmap(g, flann)
                stats.add("roadmap_test_length", i_r, roadmap_test_length)
                stats.add("roadmap_training_length", i_r,
                          roadmap_training_length)
                logger.info(f"(R) Test Length: {roadmap_test_length:.3f}")
                logger.info(
                    f"(R) Training Length: {roadmap_training_length:.3f}")

            if optimize_policy_now and optimize_poses_now:
                (general_regret, general_success, general_length
                 ) = eval.evaluate_both(policy_model, g, flann)
                stats.add("general_regret", i_r, general_regret)
                stats.add("general_success", i_r, general_success)
                stats.add("general_length", i_r, general_length)
                logger.info(f"(G) Regret: {general_regret:e}")
                logger.info(f"(G) Success: {general_success}")
                logger.info(f"(G) Length: {general_length:.3f}")

            end_eval_time = time.process_time()
            eval_time_perc = (end_eval_time - end_optimization_time) / \
                (end_eval_time - start_time)

            stats.add("general_new_data_percentage",
                      i_r, float(new_data_percentage))
            stats.add("n_policy_data_len", i_r, float(data_len))
            stats.add("general_eval_time_perc", i_r, float(eval_time_perc))
            logger.info(f"(G) New data: {new_data_percentage*100:.1f}%")
            logger.info(f"(G) Data length: {data_len}")
            logger.info(f"(G) Eval time: {eval_time_perc*100:.1f}%")

        pb.progress()
    runtime = pb.end()
    stats.add_static("runtime", str(runtime))

    # Plot stats
    if save_images:
        prefixes = ["roadmap", "policy", "general"]
        _, axs = plt.subplots(len(prefixes), 1, sharex=True,
                              figsize=(20, 30), dpi=200)
        for i_x, part in enumerate(prefixes):
            for k, v in stats.get_stats_wildcard(f"{part}.*").items():
                axs[i_x].plot(v[0], v[1], label=k)  # type: ignore
            axs[i_x].legend()  # type: ignore
            axs[i_x].xaxis.set_major_locator(  # type: ignore
                MaxNLocator(integer=True))
        plt.xlabel("Run")
        plt.savefig(f"{save_folder}/{prefix}_stats.png")

    # Save results
    if save_images:
        draw_graph(g, map_img, title="End")
        plt.savefig(f"{save_folder}/{prefix}_end.png")
    stats.to_yaml(f"{save_folder}/{prefix}_stats.yaml")
    nx.write_gpickle(g, f"{save_folder}/{prefix}_graph.gpickle")
    torch.save(policy_model.state_dict(),
               f"{save_folder}/{prefix}_policy_model.pt")

    logger.info(stats.get_statics())


if __name__ == "__main__":
    # multiprocessing
    tmp.set_sharing_strategy('file_system')
    tmp.set_start_method('spawn')

    # debug run
    for d in os.listdir("multi_optim/results/debug_data"):
        os.remove(f"multi_optim/results/debug_data/{d}")
    logging.getLogger(__name__).setLevel(logging.DEBUG)
    logging.getLogger(
        "planner.mapf_implementations.plan_cbs_roadmap"
    ).setLevel(logging.DEBUG)
    logging.getLogger(
        "sim.decentralized.policy"
    ).setLevel(logging.DEBUG)
    run_optimization(
        n_nodes=8,
        n_runs_pose=16,
        n_runs_policy=32,
        n_epochs_per_run_policy=8,
        batch_size_policy=16,
        stats_and_eval_every=8,
        lr_pos=1e-2,
        lr_policy=1e-3,
        n_agents=4,
        map_fname="roadmaps/odrm/odrm_eval/maps/x.png",
        seed=0,
        prefix="debug")

    # tiny run
    logging.getLogger(__name__).setLevel(logging.INFO)
    logging.getLogger(
        "planner.mapf_implementations.plan_cbs_roadmap"
    ).setLevel(logging.INFO)
    run_optimization(
        n_nodes=16,
        n_runs_pose=2,
        n_runs_policy=128,
        n_epochs_per_run_policy=128,
        batch_size_policy=128,
        stats_and_eval_every=2,
        lr_pos=1e-4,
        lr_policy=1e-3,
        n_agents=4,
        map_fname="roadmaps/odrm/odrm_eval/maps/x.png",
        seed=0,
        load_policy_model="multi_optim/results/tiny_model_to_load.pt",
        prefix="tiny")

    # tiny_varpose run
    logging.getLogger(__name__).setLevel(logging.INFO)
    logging.getLogger(
        "planner.mapf_implementations.plan_cbs_roadmap"
    ).setLevel(logging.INFO)
    run_optimization(
        n_nodes=16,
        n_runs_pose=64,
        n_runs_policy=128,
        n_epochs_per_run_policy=128,
        batch_size_policy=128,
        stats_and_eval_every=2,
        lr_pos=1e-4,
        lr_policy=1e-3,
        n_agents=4,
        map_fname="roadmaps/odrm/odrm_eval/maps/x.png",
        seed=0,
        load_policy_model="multi_optim/results/tiny_model_to_load.pt",
        prefix="tiny_varpose")

    # medium run
    logging.getLogger(__name__).setLevel(logging.INFO)
    logging.getLogger(
        "planner.mapf_implementations.plan_cbs_roadmap"
    ).setLevel(logging.INFO)
    run_optimization(
        n_nodes=64,
        n_runs_pose=2,
        n_runs_policy=128,
        n_epochs_per_run_policy=128,
        batch_size_policy=128,
        stats_and_eval_every=2,
        lr_pos=1e-4,
        lr_policy=1e-3,
        n_agents=4,
        map_fname="roadmaps/odrm/odrm_eval/maps/x.png",
        seed=0,
        # load_policy_model="multi_optim/results/medium_model_to_load.pt",
        prefix="medium")

    # large run
    logging.getLogger(__name__).setLevel(logging.INFO)
    logging.getLogger(
        "planner.mapf_implementations.plan_cbs_roadmap"
    ).setLevel(logging.INFO)
    run_optimization(
        n_nodes=256,
        n_runs_pose=2,
        n_runs_policy=128,
        n_epochs_per_run_policy=128,
        batch_size_policy=128,
        stats_and_eval_every=2,
        lr_pos=1e-4,
        lr_policy=1e-3,
        n_agents=4,
        map_fname="roadmaps/odrm/odrm_eval/maps/x.png",
        seed=0,
        # load_policy_model="multi_optim/results/medium_model_to_load.pt",
        prefix="large")
