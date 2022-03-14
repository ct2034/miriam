import copy
import datetime
import logging
import os
import pickle
import socket
import tracemalloc
from copy import deepcopy
from random import Random
from typing import Dict, List, Optional, Tuple
from unittest import result

import git.repo
import networkx as nx
import numpy as np
import scenarios
import scenarios.solvers
import tools
import torch
import torch.multiprocessing as tmp
from cuda_util import pick_gpu_lowest_memory
from definitions import (IDX_AVERAGE_LENGTH, IDX_SUCCESS, INVALID, MAP_IMG,
                         PATH, PATH_W_COORDS, C)
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from planner.policylearn.edge_policy import EdgePolicyDataset, EdgePolicyModel
from planner.policylearn.edge_policy_graph_utils import BFS_TYPE, TIMEOUT
from pyflann import FLANN
from roadmaps.var_odrm_torch.var_odrm_torch import (
    draw_graph, make_graph_and_flann, optimize_poses_from_paths, read_map,
    sample_points)
from sim.decentralized.agent import Agent
from sim.decentralized.iterators import IteratorType
from sim.decentralized.policy import LearnedPolicy, OptimalPolicy
from sim.decentralized.runner import run_a_scenario
from tools import ProgressBar, StatCollector
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

if __name__ == "__main__":
    from state import ScenarioState, make_a_state_with_an_upcoming_decision
else:
    from multi_optim.state import (ACTION, ScenarioState,
                                   make_a_state_with_an_upcoming_decision)

logger = logging.getLogger(__name__)

MAX_STEPS = 10
RADIUS = 0.001


def sample_trajectory_proxy(args):
    return sample_trajectory(*args)


def sample_trajectory(seed, graph, n_agents, model, map_img: MAP_IMG,
                      max_steps=MAX_STEPS):
    """Sample a trajectory using the given policy."""
    rng = Random(seed)
    starts = None
    goals = None
    flann = FLANN()
    pos = graph.nodes.data("pos")
    pos_np = np.array([pos[n] for n in graph.nodes])
    flann.build_index(np.array(pos_np))

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
                random_seed=rng.randint(0, 2**32))
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
                paths = [(starts_coord[i], goals_coord[i], state.paths_out[i])
                         for i in range(n_agents)]
                break
            observations = state.observe()
            actions: Dict[int, ACTION] = {}
            assert observations is not None, "observations is None"
            for i_a, (d, bfs) in observations.items():
                d_copy = deepcopy(d)
                del d
                # find actions to take using the policy
                actions[i_a] = model.predict(d_copy.x, d_copy.edge_index, bfs)
                # observation, action pairs for learning
                these_ds.append(d_copy)
            state.step(actions)
        except RuntimeError as e:
            logger.warning("RuntimeError: {}".format(e))
            break
    return these_ds, paths


def _get_data_folder(prefix):
    return f"multi_optim/results/{prefix}_data"


def _get_path_data(prefix, hash) -> str:
    folder = _get_data_folder(prefix)
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder+f"/{hash}.pkl"


def sample_trajectories_in_parallel(
        model: EdgePolicyModel, graph: nx.Graph, map_img: MAP_IMG, flann,
        n_agents: int, n_episodes: int, prefix: str, require_paths: bool,
        pool, rng: Random
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
    new_fname: str = _get_path_data(prefix, generation_hash)
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


def find_collisions(agents
                    ) -> Dict[Tuple[int, int], Tuple[int, int]]:
    # {(node, t): agent}
    agent_visited: Dict[Tuple[int, int], int] = {}
    # {(node, t): (agent1, agent2)}
    collisions: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for i_a, a in enumerate(agents):
        assert a.path is not None
        for t, node in enumerate(a.path):
            node_t = (node, t)
            if node_t in agent_visited.keys():
                collisions[node_t] = (
                    agent_visited[node_t],
                    i_a
                )
            agent_visited[node_t] = i_a
    return collisions


def eval_full_scenario(
    model, g: nx.Graph, n_agents, n_eval, rng
) -> Tuple[Optional[float], float, float]:
    regret_s = []
    success_s = []
    length_s = []
    model.eval()
    for i_e in range(n_eval):
        # logger.debug(f"Eval {i_e}")
        starts = rng.sample(g.nodes(), n_agents)
        goals = rng.sample(g.nodes(), n_agents)
        failed_at_creation = False

        res_policy = (0., 0., 0., 0., 0)
        res_optim = (0., 0., 0., 0., 0)
        for policy in [OptimalPolicy, LearnedPolicy]:
            agents = []
            for i_a in range(n_agents):
                a = Agent(g, starts[i_a], radius=RADIUS)
                res = a.give_a_goal(goals[i_a])
                if not res:  # failed to find a path
                    failed_at_creation = True
                a.policy = policy(a, model)
                agents.append(a)
            if not failed_at_creation:
                if policy is LearnedPolicy:
                    res_policy = run_a_scenario(
                        env=g,
                        agents=tuple(agents),
                        plot=False,
                        iterator=IteratorType.LOOKAHEAD2,
                        ignore_finished_agents=False)
                elif policy is OptimalPolicy:
                    try:
                        res_optim = run_a_scenario(
                            env=g,
                            agents=tuple(agents),
                            plot=False,
                            iterator=IteratorType.LOOKAHEAD2,
                            ignore_finished_agents=False)
                    except Exception as e:
                        logger.error(e)

        success = res_policy[IDX_SUCCESS] == 1 and res_optim[IDX_SUCCESS] == 1
        # logger.debug(f"success: {success}")
        if success:
            regret = res_policy[IDX_AVERAGE_LENGTH] - \
                res_optim[IDX_AVERAGE_LENGTH]

            if regret < 0:
                logger.warning("Regret is negative")
            #     DEBUG
            #     torch.save(model.state_dict(), "debug.pt")
            #     nx.write_gpickle(g, f"debug.gpickle")
            #     print(f"Starts: {starts}")
            #     print(f"Goals: {goals}")
            #     raise Exception("Regret is negative")

            regret_s.append(regret)
            length_s.append(res_policy[IDX_AVERAGE_LENGTH])
        success_s.append(res_policy[IDX_SUCCESS])
    if len(regret_s) > 0:
        return np.mean(regret_s), np.mean(success_s), np.mean(length_s)
    else:
        return None, np.mean(success_s), np.mean(length_s)


def make_eval_set(model, g: nx.Graph, n_agents, n_eval, rng
                  ) -> List[Tuple[Data, BFS_TYPE]]:
    model.eval()
    eval_set = []
    for i_e in range(n_eval):
        state = make_a_state_with_an_upcoming_decision(
            g, n_agents,  model, RADIUS, rng)
        observation = state.observe()
        assert observation is not None
        i_a = rng.sample(list(observation.keys()), 1)[0]
        data, big_from_small = observation[i_a]
        eval_set.append((data, big_from_small))
    return eval_set


def optimize_policy(model, batch_size, optimizer, epds
                    ) -> Tuple[EdgePolicyModel, float]:
    loader = DataLoader(epds, num_workers=0,
                        batch_size=batch_size, shuffle=True)
    loss_s = []
    for _, batch in enumerate(loader):
        loss = model.learn(batch, optimizer)
        loss_s.append(loss)

    if len(loss_s) == 0:
        loss_s = [0]
    return model, np.mean(loss_s)


def run_optimization(
        n_nodes: int = 32,
        n_runs_pose: int = 1024,
        n_runs_policy: int = 128,
        n_epochs_per_run_policy: int = 64,
        batch_size_policy: int = 32,
        stats_and_eval_every: int = 1,
        lr_pos: float = 1e-4,
        lr_policy: float = 4e-4,
        n_agents: int = 8,
        map_fname: str = "roadmaps/odrm/odrm_eval/maps/x.png",
        load_policy_model: Optional[str] = None,
        rng: Random = Random(0),
        prefix: str = "noname"):
    logger.info("run_optimization")
    torch.manual_seed(rng.randint(0, 2 ** 32))

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
    policy_eval_list = make_eval_set(
        policy_model, g, n_agents, 100, rng)

    # Data for policy
    epds = EdgePolicyDataset(f"multi_optim/results/{prefix}_data")

    # Visualization and analysis
    stats = StatCollector([
        # "poses_test_length",
        "poses_training_length",
        "policy_loss",
        "policy_regret",
        "policy_success",
        "policy_accuracy",
        "general_new_data_percentage",
        "general_length",
        "n_policy_data_len"])
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
    draw_graph(g, map_img, title="Start")
    plt.savefig(f"multi_optim/results/{prefix}_start.png")

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
    # poses_test_length = 0
    poses_training_length = 0
    for i_r in range(n_runs):
        optimize_poses_now: bool = i_r % n_runs_per_run_pose == 0
        optimize_policy_now: bool = i_r % n_runs_per_run_policy == 0

        # Sample runs for both optimizations
        assert n_runs_policy >= n_runs_pose, \
            "otherwise we dont need optiomal solution that often"
        old_data_len = len(epds)
        new_fname, paths_s = sample_trajectories_in_parallel(
            policy_model, g, map_img, flann, n_agents, n_epochs_per_run_policy,
            prefix, optimize_poses_now, pool, rng)
        epds.add_file(new_fname)
        data_len = len(epds)
        new_data_percentage = (data_len - old_data_len) / data_len

        # Optimizing Poses
        if optimize_poses_now:
            (g, pos, flann, poses_training_length
             ) = optimize_poses_from_paths(
                g, pos, paths_s, map_img, optimizer_pos)
            if i_r % stats_and_eval_every == 0:
                # stats.add("poses_test_length", i_r, float(poses_test_length))
                stats.add("poses_training_length", i_r,
                          float(poses_training_length))

        # Optimizing Policy
        if optimize_policy_now:
            policy_model, policy_loss = optimize_policy(
                policy_model, batch_size_policy, optimizer_policy, epds)
            if i_r % stats_and_eval_every == 0:
                # also eval now
                rng_test = Random(1)
                n_eval = 10
                # little less agents for evaluation
                eval_n_agents = int(np.ceil(n_agents * .7))
                (regret, success, general_length) = eval_full_scenario(
                    policy_model, g, eval_n_agents, n_eval, rng_test)
                policy_accuracy = policy_model.accuracy(policy_eval_list)

                stats.add("policy_loss", i_r, float(policy_loss))
                if regret is not None:
                    stats.add("policy_regret", i_r, float(regret))
                stats.add("policy_success", i_r, float(success))
                stats.add("policy_accuracy", i_r, policy_accuracy)
                stats.add("general_length", i_r, float(general_length))
                stats.add("general_new_data_percentage",
                          i_r, float(new_data_percentage))
                stats.add("n_policy_data_len", i_r, float(data_len))
                logger.info(f"Loss: {policy_loss:.3f}")
                logger.info(f"Regret: {regret:e}")
                logger.info(f"Success: {success}")
                logger.info(f"Accuracy: {policy_accuracy:.3f}")
                logger.info(f"Length: {general_length:.3f}")
                logger.info(f"New data: {new_data_percentage*100:.1f}%")
                logger.info(f"Data length: {data_len}")

        pb.progress()
    runtime = pb.end()
    stats.add_static("runtime", str(runtime))

    # Plot stats
    prefixes = ["poses", "policy", "general"]
    _, axs = plt.subplots(len(prefixes), 1, sharex=True,
                          figsize=(20, 30), dpi=200)
    for i_x, part in enumerate(prefixes):
        for k, v in stats.get_stats_wildcard(f"{part}.*").items():
            axs[i_x].plot(v[0], v[1], label=k)  # type: ignore
        axs[i_x].legend()  # type: ignore
        axs[i_x].xaxis.set_major_locator(  # type: ignore
            MaxNLocator(integer=True))
    plt.xlabel("Run")
    plt.savefig(f"multi_optim/results/{prefix}_stats.png")

    # Save results
    draw_graph(g, map_img, title="End")
    plt.savefig(f"multi_optim/results/{prefix}_end.png")
    stats.to_yaml(f"multi_optim/results/{prefix}_stats.yaml")
    nx.write_gpickle(g, f"multi_optim/results/{prefix}_graph.gpickle")
    torch.save(policy_model.state_dict(),
               f"multi_optim/results/{prefix}_policy_model.pt")

    logger.info(stats.get_statics())


if __name__ == "__main__":
    tracemalloc.start()

    # multiprocessing
    tmp.set_sharing_strategy('file_system')
    tmp.set_start_method('spawn')

    # debug run
    for d in os.listdir("multi_optim/results/debug_data"):
        os.remove(f"multi_optim/results/debug_data/{d}")
    rng = Random(0)
    logging.getLogger(__name__).setLevel(logging.DEBUG)
    logging.getLogger(
        "planner.mapf_implementations.plan_cbs_roadmap"
    ).setLevel(logging.DEBUG)
    logging.getLogger(
        "sim.decentralized.policy"
    ).setLevel(logging.DEBUG)
    run_optimization(
        n_nodes=8,
        n_runs_pose=2,
        n_runs_policy=16,
        n_epochs_per_run_policy=4,
        batch_size_policy=16,
        stats_and_eval_every=1,
        lr_pos=1e-4,
        lr_policy=1e-3,
        n_agents=4,
        map_fname="roadmaps/odrm/odrm_eval/maps/x.png",
        rng=rng,
        prefix="debug")

    # tiny run
    rng = Random(0)
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
        load_policy_model="multi_optim/results/tiny_model_to_load.pt",
        rng=rng,
        prefix="tiny")

    # tiny_varpose run
    rng = Random(0)
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
        load_policy_model="multi_optim/results/tiny_model_to_load.pt",
        rng=rng,
        prefix="tiny_varpose")

    # medium run
    rng = Random(0)
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
        # load_policy_model="multi_optim/results/medium_model_to_load.pt",
        rng=rng,
        prefix="medium")

    # large run
    rng = Random(0)
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
        # load_policy_model="multi_optim/results/medium_model_to_load.pt",
        rng=rng,
        prefix="large")
