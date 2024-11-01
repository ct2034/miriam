#!/usr/bin/env python3

import copy
import datetime
import logging
import os
import pickle
import socket
import time
from math import ceil
from random import Random
from typing import Dict, List, Optional, Tuple

import cv2
import git.repo
import networkx as nx
import numpy as np
import torch
import torch.multiprocessing as tmp
import wandb
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from pyflann import FLANN
from torch_geometric.loader import DataLoader

import scenarios
import scenarios.solvers
import tools
from cuda_util import pick_gpu_lowest_memory
from definitions import DEFAULT_TIMEOUT_S, INVALID, MAP_IMG, PATH_W_COORDS, POS
from multi_optim.configs import configs_more_lr_pos_s
from planner.policylearn.edge_policy import EdgePolicyDataset, EdgePolicyModel
from roadmaps.reaction_diffusion.example_figure import make_fig
from roadmaps.reaction_diffusion.rd import sample_points_reaction_diffusion
from roadmaps.var_odrm_torch.var_odrm_torch import (
    draw_graph,
    make_graph_and_flann,
    optimize_poses_from_paths,
    read_map,
    sample_points,
)
from scenarios.generators import movingai_read_mapfile
from tools import ProgressBar, StatCollector

if __name__ == "__main__":
    from configs import configs_all_maps
    from eval import Eval
    from state import ACTION, ITERATOR_TYPE, ScenarioState
else:
    from multi_optim.configs import configs_all_maps
    from multi_optim.eval import Eval
    from multi_optim.state import ACTION, ITERATOR_TYPE, ScenarioState

logger = logging.getLogger(__name__)

MAX_STEPS = 10


def sample_trajectory_proxy(args):
    return sample_trajectory(*args)


def sample_trajectory(
    seed: int,
    graph: nx.Graph,
    n_agents: int,
    model: EdgePolicyModel,
    map_img: MAP_IMG,
    radius: float,
    max_steps: int = MAX_STEPS,
):
    """Sample a trajectory using the given policy."""
    start_time = time.process_time()
    rng = Random(seed)
    starts = None
    goals = None
    flann = FLANN()
    pos = nx.get_node_attributes(graph, POS)
    pos_np = np.array([pos[n] for n in graph.nodes])
    flann.build_index(np.array(pos_np, dtype=np.float32), random_index=0)
    model.train()

    starts_coord: Optional[List[Tuple[float, float]]] = None
    goals_coord: Optional[List[Tuple[float, float]]] = None

    solvable = False
    while not solvable:
        unique = False
        while not unique:
            starts_goals_coord = sample_points(n_agents * 2, map_img, rng)
            result, _ = flann.nn_index(
                starts_goals_coord.detach().numpy(), 1, random_seed=0
            )
            starts_coord = [
                starts_goals_coord[i, :2].detach().numpy().astype(float)
                for i in range(n_agents)
            ]
            goals_coord = [
                starts_goals_coord[i, :2].detach().numpy().astype(float)
                for i in range(n_agents, n_agents * 2)
            ]
            starts = result[0:n_agents].tolist()
            goals = result[n_agents:].tolist()
            unique = len(set(starts)) == n_agents and len(set(goals)) == n_agents
        # is this solvable?
        optimal_paths = scenarios.solvers.cached_cbsr(
            graph, starts, goals, radius=radius, timeout=int(DEFAULT_TIMEOUT_S * 0.9)
        )
        solvable = optimal_paths != INVALID

    assert starts_coord is not None
    assert goals_coord is not None

    state = ScenarioState(graph, starts, goals, model, radius)
    state.run()

    # Sample states
    these_ds = []
    paths = None  # type: Optional[List[PATH_W_COORDS]]
    for i_s in range(max_steps):
        try:
            if state.finished:
                if len(state.paths_out[0]) > 0:
                    paths = [
                        (starts_coord[i], goals_coord[i], state.paths_out[i])
                        for i in range(n_agents)
                    ]
                else:
                    paths = None
                break
            observations = state.observe()
            actions: Dict[int, ACTION] = {}
            assert observations is not None, "observations is None"
            for i_a, (d, bfs) in observations.items():
                # find actions to take using the policy
                actions[i_a] = model.predict_probablilistic(d.x, d.edge_index, bfs)
                # observation, action pairs for learning
                these_ds.append(d)
            state.step(actions)
        except RuntimeError as e:
            logger.warning("RuntimeError: {}".format(e))
            break
    return these_ds, paths, time.process_time() - start_time


def _get_data_folder(save_folder, prefix):
    return f"{save_folder}/{prefix}_data"


def _get_path_data(save_folder, prefix, hash) -> str:
    folder = _get_data_folder(save_folder, prefix)
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder + f"/{hash}.pkl"


def write_stats_png(prefix, save_folder, stats):
    prefixes = [
        "roadmap",
        "policy_accuracy",
        "policy_regret",
        "policy_success",
        # "general_accuracy",
        "general_length",
        "general_regret",
        "general_success",
        # "general_gen"
        # "runtime_",
        "data_",
    ]
    _, axs = plt.subplots(
        len(prefixes), 1, sharex=True, figsize=(10, 6 * len(prefixes)), dpi=200
    )
    for i_x, part in enumerate(prefixes):
        subset = stats.get_stats_wildcard(f"{part}.*")
        keys = subset.keys()
        for k in sorted(keys):
            v = subset[k]
            axs[i_x].plot(v[0], v[1], label=k)  # type: ignore
        axs[i_x].legend()  # type: ignore
        axs[i_x].xaxis.set_major_locator(MaxNLocator(integer=True))  # type: ignore
    plt.xlabel(f"Run {prefix}")
    plt.tight_layout()
    plt.savefig(f"{save_folder}/{prefix}_stats.png")


def sample_trajectories_in_parallel(
    model: EdgePolicyModel,
    graph: nx.Graph,
    map_img: MAP_IMG,
    radius: float,
    _,
    n_agents: int,
    n_episodes: int,
    prefix: str,
    require_paths: bool,
    save_folder,
    pool,
    rng: Random,
) -> Tuple[str, List[List[PATH_W_COORDS]], float, float]:
    model_copy = EdgePolicyModel()
    model_copy.load_state_dict(copy.deepcopy(model.state_dict()))
    model_copy.eval()

    params = [
        (s, graph, n_agents, model_copy, map_img, radius)
        for s in rng.sample(range(2**32), k=n_episodes)
    ]
    generation_hash = tools.hasher(
        [],
        {
            "seeds": [p[0] for p in params],
            "graph": graph,
            "n_agents": n_agents,
            "model": model_copy,
        },
    )
    new_fname: str = _get_path_data(save_folder, prefix, generation_hash)
    # only create file if this data does not exist or if paths are required
    paths_s: List[List[PATH_W_COORDS]] = []
    ts: List[float] = []
    if os.path.exists(new_fname) and not require_paths:
        pass
    else:
        results_s = pool.imap_unordered(sample_trajectory_proxy, params)
        # results_s = map(sample_trajectory_proxy, params)
        new_ds = []
        for ds, paths, t in results_s:
            new_ds.extend(ds)
            ts.append(t)
            if paths is not None:
                paths_s.append(paths)
        with open(new_fname, "wb") as f:
            pickle.dump(new_ds, f)

    # runtime stats
    if len(ts) > 0:
        ts_max: float = max(ts)
        ts_mean: float = float(np.mean(ts))
    else:
        ts_max = 0.0
        ts_mean = 0.0

    return new_fname, paths_s, ts_max, ts_mean


def optimize_policy(
    model, batch_size, optimizer, epds, n_epochs
) -> Tuple[EdgePolicyModel, float]:
    if len(epds) == 0:
        return model, 0.0
    loader = DataLoader(epds, batch_size=batch_size, shuffle=True)
    loss_s = []
    for _ in range(n_epochs):
        for _, batch in enumerate(loader):
            loss = model.learn(batch, optimizer)
            loss_s.append(loss)
        loader.dataset.shuffle()

    if len(loss_s) == 0:
        loss_s = [0]
    return model, float(np.mean(loss_s))


def run_optimization(
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
    radius: float = 0.001,
    use_reaction_diffusion: bool = True,
    load_policy_model: Optional[str] = None,
    load_roadmap: Optional[str] = None,
    prefix: str = "noname",
    save_images: bool = True,
    save_folder: Optional[str] = None,
    pool_in: Optional[tmp.Pool] = None,
):  # type: ignore
    wandb_run = wandb.init(
        project="miriam-multi-optim-run",
        name=f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}" + f"-{prefix}",
        reinit=True,
        settings=wandb.Settings(start_method="fork"),
    )
    assert wandb_run is not None

    rng = Random(seed)
    logger.info(f"run_optimization {prefix}")
    torch.manual_seed(rng.randint(0, 2**32))
    if save_folder is None:
        save_folder = "multi_optim/results"  # default

    # multiprocessing
    if pool_in is None:
        n_processes = min(tmp.cpu_count(), 16)
        pool = tmp.Pool(processes=n_processes)
    else:
        pool = pool_in

    # n_agents
    n_agents_s = list(range(2, max_n_agents + 1, 2))  # e.g. 2, 4, 6, ...
    i_n_agents: int = 0
    n_agents: int = n_agents_s[i_n_agents]

    # Roadmap
    SUPER_RES_MULTIPLIER = 3
    map_fname = f"roadmaps/odrm/odrm_eval/maps/{map_name}"
    if os.path.splitext(map_fname)[1] == ".png":
        map_img: MAP_IMG = read_map(map_fname)
        map_img_np = np.array(map_img, dtype=np.uint8)
    elif os.path.splitext(map_fname)[1] == ".map":
        map_img_np = movingai_read_mapfile(map_fname)
        map_img = gridmap_to_map_img(map_img_np)
    else:
        raise ValueError(f"Unknown map file extension {map_fname}")
    map_ocv = cv2.Mat(map_img_np)
    map_img_inflated = cv2.erode(
        map_ocv, np.ones((SUPER_RES_MULTIPLIER, SUPER_RES_MULTIPLIER), np.uint8)
    )

    B = None
    if load_roadmap is not None:
        # load graph from file
        graph_loaded = nx.read_gpickle(load_roadmap)
        assert isinstance(graph_loaded, nx.Graph)
        pos_dict = nx.get_node_attributes(graph_loaded, POS)
        pos = torch.tensor(
            np.array([pos_dict[k] for k in graph_loaded.nodes()]),
            device=torch.device("cpu"),
            dtype=torch.float,
            requires_grad=True,
        )
    else:
        if not use_reaction_diffusion:
            pos = sample_points(n_nodes, map_img_inflated, rng)
        else:  # use reaction diffusion
            pos, B = sample_points_reaction_diffusion(n_nodes, map_img, rng)
    optimizer_pos = torch.optim.Adam([pos], lr=lr_pos)
    g: nx.Graph
    flann: FLANN
    (g, flann) = make_graph_and_flann(pos, map_img_inflated, n_nodes, rng)
    if save_images:
        draw_graph(g, map_img, title="Start")
        plt.savefig(f"{save_folder}/{prefix}_start.png")
    if save_images and B is not None:
        make_fig(
            cv2.rotate(B, cv2.ROTATE_90_COUNTERCLOCKWISE),
            len(map_img) / 256,
            map_img,
            fname=f"{save_folder}/{prefix}_start_rd.png",
        )

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
    optimizer_policy = torch.optim.Adam(policy_model.parameters(), lr=lr_policy)
    policy_loss: Optional[float] = None

    # Eval
    # little less agents for evaluation
    eval = Eval(
        g,
        map_img,
        n_agents_s=n_agents_s,
        n_eval_per_n_agents=8,
        iterator_type=ITERATOR_TYPE,
        radius=radius,
    )

    # Data for policy
    clear_data_folder(prefix, save_folder)
    epds = EdgePolicyDataset(f"{save_folder}/{prefix}_data")

    # Visualization and analysis
    static_stats = {
        # metadata
        "hostname": socket.gethostname(),
        "git_hash": git.repo.Repo(".").head.object.hexsha,
        "started_at": datetime.datetime.now().isoformat(),
        "gpu": str(gpu),
        # parameters
        "n_nodes": n_nodes,
        "n_runs_pose": n_runs_pose,
        "n_runs_policy": n_runs_policy,
        "n_episodes_per_run_policy": n_episodes_per_run_policy,
        "n_epochs_per_run_policy": n_epochs_per_run_policy,
        "batch_size_policy": batch_size_policy,
        "stats_every": stats_and_eval_every,
        "lr_pos": lr_pos,
        "lr_policy": lr_policy,
        "max_n_agents": max_n_agents,
        "map_fname": map_fname,
        "radius": radius,
        "use_reaction_diffusion": use_reaction_diffusion,
        "load_policy_model": (load_policy_model if load_policy_model else "None"),
        "prefix": prefix,
    }
    stats = StatCollector()
    for stat_saver in [wandb.config.update, stats.add_statics]:
        stat_saver(static_stats)

    # Making sense of two n_runs
    n_runs = max(n_runs_pose, n_runs_policy)
    if n_runs_policy > n_runs_pose:
        n_runs_per_run_policy = 1
        if n_runs_pose > 0:
            n_runs_per_run_pose = n_runs // n_runs_pose
        else:
            n_runs_per_run_pose = 0
    else:  # n_runs_pose > n_runs_policy
        n_runs_per_run_pose = 1
        if n_runs_policy > 0:
            n_runs_per_run_policy = n_runs // n_runs_policy
        else:
            n_runs_per_run_policy = 0

    # Run optimization
    pb = ProgressBar(
        name=f"{prefix} Optimization",
        total=n_runs + 1,
        step_perc=1,
        print_func=logger.info,
    )
    # roadmap_test_length = 0
    roadmap_training_length = 0
    for i_r in range(0, n_runs + 1):
        wandb.log({"general/progress": float(i_r) / n_runs}, step=i_r)
        start_time = time.process_time()
        if n_runs_per_run_pose > 0:
            optimize_poses_now: bool = i_r % n_runs_per_run_pose == 0
        else:
            optimize_poses_now = False
        if n_runs_per_run_policy > 0:
            optimize_policy_now: bool = i_r % n_runs_per_run_policy == 0
        else:
            optimize_policy_now = False

        # n_agents
        switching_metric_str = f"policy_accuracy_{n_agents}"
        switching_metric: float = 0.0
        try:
            switching_metric_stats = stats.get_stats(switching_metric_str)
            switching_metric = switching_metric_stats[switching_metric_str][1][
                -1
            ]  # type: ignore
        except KeyError:
            # in case we don't have any data yet
            pass
        if switching_metric >= 0.7:
            i_n_agents = min(i_n_agents + 1, len(n_agents_s) - 1)
        n_agents = n_agents_s[i_n_agents]

        # Sample runs for both optimizations
        # assert n_runs_policy >= n_runs_pose, \
        #     "otherwise we dont need optiomal solution that often"
        old_data_len = len(epds)
        new_fname, paths_s, ts_max, ts_mean = sample_trajectories_in_parallel(
            policy_model,
            g,
            map_img,
            radius,
            flann,
            n_agents,
            n_episodes_per_run_policy,
            prefix,
            optimize_poses_now,
            save_folder,
            pool,
            rng,
        )
        epds.add_file(new_fname)
        data_len = len(epds)
        if data_len > 0:
            new_data_percentage = (data_len - old_data_len) / data_len
        else:
            new_data_percentage = 0.0
        end_time_generation = time.process_time()
        stats.add("runtime_generation_all", i_r, (end_time_generation - start_time))
        wandb.log(
            {"runtime/generation/all": (end_time_generation - start_time)}, step=i_r
        )

        # Optimizing Poses
        if optimize_poses_now:
            (g, pos, flann, roadmap_training_length) = optimize_poses_from_paths(
                g, pos, paths_s, map_img_inflated, n_nodes, optimizer_pos, rng
            )
            end_time_optim_poses = time.process_time()
            stats.add(
                "runtime_optim_poses", i_r, (end_time_optim_poses - end_time_generation)
            )
            wandb.log(
                {"runtime/optim/poses": (end_time_optim_poses - end_time_generation)},
                step=i_r,
            )
            stats.add("roadmap_nodes_precentage", i_r, (len(g.nodes) / n_nodes))
            wandb.log({"roadmap/nodes_precentage": (len(g.nodes) / n_nodes)}, step=i_r)

        # Optimizing Policy
        if optimize_policy_now:
            start_time_optim_policy = time.process_time()
            policy_model, policy_loss = optimize_policy(
                policy_model,
                batch_size_policy,
                optimizer_policy,
                epds,
                n_epochs_per_run_policy,
            )
            end_time_optim_policy = time.process_time()
            stats.add(
                "runtime_optim_policy",
                i_r,
                (end_time_optim_policy - start_time_optim_policy),
            )
            wandb.log(
                {
                    "runtime/optim/policy": (
                        end_time_optim_policy - start_time_optim_policy
                    )
                },
                step=i_r,
            )

        if i_r % stats_and_eval_every == 0:
            end_optimization_time = time.process_time()

            if optimize_policy_now or i_r == 0:
                # also eval now
                policy_results = eval.evaluate_policy(policy_model)
                names = sorted(policy_results.keys())
                for name in names:
                    stats.add(f"policy_{name}", i_r, policy_results[name])
                    logger.info(f"(P) {name}: {policy_results[name]}")
                    wandb.log({f"policy/{name}": policy_results[name]}, step=i_r)
                if policy_loss is not None:
                    stats.add("policy_loss", i_r, policy_loss)
                    wandb.log({"policy/loss": policy_loss}, step=i_r)

            if optimize_poses_now or i_r == 0:
                # eval the current roadmap
                roadmap_test_length = eval.evaluate_roadmap(g, flann)
                stats.add("roadmap_test_length", i_r, roadmap_test_length)
                stats.add("roadmap_training_length", i_r, roadmap_training_length)
                logger.info(f"(R) Test Length: {roadmap_test_length:.3f}")
                logger.info(f"(R) Training Length: {roadmap_training_length:.3f}")
                wandb.log(
                    {
                        "roadmap/test_length": roadmap_test_length,
                        "roadmap/training_length": roadmap_training_length,
                    },
                    step=i_r,
                )

            if optimize_policy_now or optimize_poses_now or i_r == 0:
                general_results = eval.evaluate_both(policy_model, g, flann)
                names = sorted(general_results.keys())
                for name in names:
                    stats.add(f"general_{name}", i_r, general_results[name])
                    logger.info(f"(G) {name}: {general_results[name]}")
                    wandb.log({f"general/{name}": general_results[name]}, step=i_r)

            end_eval_time = time.process_time()
            eval_time_perc = (end_eval_time - end_optimization_time) / (
                end_eval_time - start_time
            )
            stats.add("runtime_eval", i_r, (end_eval_time - end_optimization_time))
            stats.add("runtime_full", i_r, (end_eval_time - start_time))
            wandb.log(
                {
                    "runtime/eval": end_eval_time - end_optimization_time,
                    "runtime/full": end_eval_time - start_time,
                },
                step=i_r,
            )

            stats.add("data_len", i_r, float(data_len))
            stats.add("general_new_data_percentage", i_r, float(new_data_percentage))
            stats.add(
                "general_generation_n_agents_percentage",
                i_r,
                float(n_agents / max_n_agents),
            )
            stats.add("runtime_eval_time_perc", i_r, float(eval_time_perc))
            stats.add("runtime_generation_mean", i_r, ts_mean)
            stats.add("runtime_generation_max", i_r, ts_max)
            logger.info(f"(G) New data: {new_data_percentage*100:.1f}%")
            logger.info(f"(G) Data length: {data_len}")
            logger.info(f"(G) Eval time: {eval_time_perc*100:.1f}%")
            logger.info(f"(G) Generation n_agents: {n_agents}")
            logger.info(f"(G) Runtime generation mean: {ts_mean:.3f}s")
            logger.info(f"(G) Runtime generation max: {ts_max:.3f}s")
            wandb.log(
                {
                    "general/data_len": data_len,
                    "general/new_data_percentage": new_data_percentage,
                    "general/generation_n_agents_percentage": (n_agents / max_n_agents),
                    "runtime/eval_time/perc": eval_time_perc,
                    "runtime/generation/mean": ts_mean,
                    "runtime/generation/max": ts_max,
                },
                step=i_r,
            )

        pb.progress()
    runtime = pb.end()
    stats.add_static("runtime", str(runtime))
    if pool_in is None:
        # we made our own pool, so we need to close it
        pool.close()
        pool.terminate()

    # Plot stats
    if save_images:
        write_stats_png(prefix, save_folder, stats)

    # Save results
    if save_images:
        draw_graph(g, map_img, title="End")
        plt.savefig(f"{save_folder}/{prefix}_end.png")
    stats.to_yaml(f"{save_folder}/{prefix}_stats.yaml")
    nx.write_gpickle(g, f"{save_folder}/{prefix}_graph.gpickle")
    torch.save(policy_model.state_dict(), f"{save_folder}/{prefix}_policy_model.pt")

    wandb_run.finish()
    logger.info(stats.get_statics())


def inflate_map(map_np, SUPER_RES_MULTIPLIER):
    map_np_superres = np.repeat(
        np.repeat(map_np, SUPER_RES_MULTIPLIER, axis=0), SUPER_RES_MULTIPLIER, axis=1
    )
    map_np_inflated = np.zeros_like(map_np_superres)
    for x in range(map_np_inflated.shape[0]):
        for y in range(map_np_inflated.shape[1]):
            potential = map_np_superres[x, y]
            if x > 0:
                potential = max(potential, map_np_superres[x - 1, y])
            if x < map_np_inflated.shape[0] - 1:
                potential = max(potential, map_np_superres[x + 1, y])
            if y > 0:
                potential = max(potential, map_np_superres[x, y - 1])
            if y < map_np_inflated.shape[1] - 1:
                potential = max(potential, map_np_superres[x, y + 1])
            map_np_inflated[x, y] = potential

    return map_np_inflated


def inflate_map_img(map_img: MAP_IMG, radius: float):
    width = len(map_img[0])
    radius_px = int(ceil(radius * width))
    map_img_inflated = np.zeros_like(map_img, dtype=np.uint8)
    for x in range(map_img_inflated.shape[0]):
        for y in range(map_img_inflated.shape[1]):
            potential = map_img[x][y]
            for x2 in range(max(0, x - radius_px), min(width, x + radius_px)):
                for y2 in range(max(0, y - radius_px), min(width, y + radius_px)):
                    potential = min(potential, map_img[x2][y2])
            map_img_inflated[x, y] = potential
    return map_img_inflated


def gridmap_to_map_img(map_np: np.ndarray) -> MAP_IMG:
    map_img = tuple(((map_np - 1) * -255).tolist())
    return map_img


def clear_data_folder(prefix, save_folder):
    data_folder = f"{save_folder}/{prefix}_data"
    if os.path.exists(data_folder):
        for f in os.listdir(data_folder):
            os.remove(os.path.join(data_folder, f))


if __name__ == "__main__":
    # multiprocessing
    # tmp.set_sharing_strategy('file_system')
    tmp.set_start_method("spawn")
    # set_ulimit()  # fix `RuntimeError: received 0 items of ancdata`
    n_processes = min(tmp.cpu_count(), 16)
    pool = tmp.Pool(processes=n_processes)

    configs_to_run = {
        k: v
        for k, v in configs_more_lr_pos_s.items()
        if k.startswith("debug")
        # or k.startswith("large")
    }

    prefixes_to_run = sorted(configs_to_run.keys())
    print(f"Running {len(prefixes_to_run)} configs: {prefixes_to_run}")

    for prefix in prefixes_to_run:
        if prefix.startswith("debug"):
            level = logging.DEBUG
        else:
            level = logging.INFO
        for set_fun in [
            logging.getLogger(__name__).setLevel,
            logging.getLogger("planner.mapf_implementations.plan_cbs_roadmap").setLevel,
            logging.getLogger("sim.decentralized.policy").setLevel,
        ]:
            set_fun(level)

        # start the actual run
        run_optimization(**configs_to_run[prefix], pool_in=pool)

    pool.close()
    pool.terminate()
