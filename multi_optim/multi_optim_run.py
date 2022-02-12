import datetime
import logging
import socket
import tracemalloc
from fileinput import filename
from multiprocessing.spawn import import_main_path
from random import Random, getstate
from typing import Dict, List, Optional, Tuple

import git
import networkx as nx
import numpy as np
import torch
import torch.multiprocessing as tmp
from definitions import INVALID, SCENARIO_RESULT
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from planner.policylearn.edge_policy import EdgePolicyModel
from planner.policylearn.edge_policy_graph_utils import (BFS_TYPE, RADIUS,
                                                         get_optimal_edge)
from roadmaps.var_odrm_torch.var_odrm_torch import (draw_graph, make_graph,
                                                    optimize_poses, read_map,
                                                    sample_points)
from sim.decentralized.agent import Agent, env_to_nx
from sim.decentralized.iterators import IteratorType
from sim.decentralized.policy import EdgePolicy, OptimalEdgePolicy
from sim.decentralized.runner import run_a_scenario
from tools import ProgressBar, StatCollector
from torch_geometric.data import Data

if __name__ == "__main__":
    from dagger import DaggerStrategy, make_a_state_with_an_upcoming_decision
else:
    from multi_optim.dagger import (DaggerStrategy,
                                    make_a_state_with_an_upcoming_decision)

logger = logging.getLogger(__name__)


def find_collisions(agents
                    ) -> Dict[Tuple[int, int], Tuple[int, int]]:
    # {(node, t): agent}
    agent_visited: Dict[Tuple[int, int], int] = {}
    # {(node, t): (agent1, agent2)}
    collisions: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for i_a, a in enumerate(agents):
        assert a.path is not None
        for node in a.path:
            if node in agent_visited.keys():
                collisions[node] = (
                    agent_visited[node],
                    i_a
                )
            agent_visited[node] = i_a
    return collisions


def eval_policy_full_scenario(
    model, g: nx.Graph, env_nx: nx.Graph, n_agents, n_eval, rng
) -> Tuple[Optional[float], float]:
    regret_s = []
    success_s = []
    model.eval()
    for i_e in range(n_eval):
        # logger.debug(f"Eval {i_e}")
        starts = rng.sample(g.nodes(), n_agents)
        goals = rng.sample(g.nodes(), n_agents)
        failed_at_creation = False
        for policy in [OptimalEdgePolicy, EdgePolicy]:
            agents = []
            for i_a in range(n_agents):
                a = Agent(g, starts[i_a], env_nx=env_nx, radius=RADIUS)
                res = a.give_a_goal(goals[i_a])
                if not res:  # failed to find a path
                    failed_at_creation = True
                a.policy = policy(a, model)
                agents.append(a)
            if failed_at_creation:
                res_policy = (0., 0., 0., 0., 0.)
                res_optim = (0., 0., 0., 0., 0.)
            else:
                if policy is EdgePolicy:
                    res_policy = run_a_scenario(
                        env=g,
                        agents=tuple(agents),
                        plot=False,
                        iterator=IteratorType.EDGE_POLICY2)
                elif policy is OptimalEdgePolicy:
                    try:
                        res_optim = run_a_scenario(
                            env=g,
                            agents=tuple(agents),
                            plot=False,
                            iterator=IteratorType.EDGE_POLICY2)
                    except Exception as e:
                        logger.error(e)
                        res_optim = (0, 0, 0, 0, 0)

        success = res_policy[4] and res_optim[4]
        # logger.debug(f"success: {success}")
        if success:
            regret = res_policy[0] - res_optim[0]
            regret_s.append(regret)
            # logger.debug(f"regret: {regret}")
        success_s.append(res_policy[4])
    if len(regret_s) > 0:
        return np.mean(regret_s), np.mean(success_s)
    else:
        return None, np.mean(success_s)


def make_eval_set(model, g: nx.Graph, n_agents, n_eval, rng
                  ) -> List[Tuple[Data, BFS_TYPE]]:
    model.eval()
    env_nx = env_to_nx(g)
    eval_set = []
    for i_e in range(n_eval):
        state = make_a_state_with_an_upcoming_decision(
            g, n_agents, env_nx, model, rng)
        observation = state.observe()
        i_a = rng.sample(list(observation.keys()), 1)[0]
        data, big_from_small = observation[i_a]
        eval_set.append((data, big_from_small))
    return eval_set


def optimize_policy(model, g: nx.Graph, n_agents, n_epochs, batch_size,
                    optimizer, data_files, pool, prefix, rng):
    ds = DaggerStrategy(
        model, g, n_epochs, n_agents, batch_size, optimizer, prefix, rng)
    model, loss, new_data_percentage, data_files, data_len = ds.run_dagger(
        pool, data_files)
    return model, ds.env_nx, loss, new_data_percentage, data_files, data_len


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
        rng: Random = Random(0),
        prefix: str = "noname"):
    logger.info("run_optimization")
    torch.manual_seed(rng.randint(0, 2 ** 32))

    # multiprocessing
    n_processes = min(tmp.cpu_count(), 8)
    pool = tmp.Pool(processes=n_processes)

    # Roadmap
    map_img = read_map(map_fname)
    pos = sample_points(n_nodes, map_img, rng)
    optimizer_pos = torch.optim.Adam([pos], lr=lr_pos)
    g = make_graph(pos, map_img)

    # GPU or CPU?
    if torch.cuda.is_available():
        assert 1 == torch.cuda.device_count(),\
            "Make sure this can only see one cuda device."
        logger.info("Using GPU")
        gpu = torch.device("cuda:0")
        torch.cuda.empty_cache()
        # print(torch.cuda.memory_summary())
        # torch.cuda.set_per_process_memory_fraction(fraction=.1)
        # print(torch.cuda.memory_summary())
    else:
        logger.warning("GPU not available, using CPU")
        gpu = torch.device("cpu")

    # Policy
    policy_model = EdgePolicyModel(gpu=gpu)
    for param in policy_model.parameters():
        param.share_memory_()
    policy_model.share_memory()
    # policy_model.use_multiprocessing = False
    # policy_model.share_memory()
    optimizer_policy = torch.optim.Adam(
        policy_model.parameters(), lr=lr_policy)
    policy_data_files = []  # type: List[str]
    policy_eval_list = make_eval_set(
        policy_model, g, n_agents, 100, rng)

    # Visualization and analysis
    stats = StatCollector([
        "poses_test_length",
        "poses_training_length",
        "policy_loss",
        "policy_regret",
        "policy_success",
        "policy_accuracy",
        "policy_new_data_percentage",
        "n_policy_data_len"])
    stats.add_statics({
        # metadata
        "hostname": socket.gethostname(),
        "git_hash": git.Repo(".").head.object.hexsha,
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
    for i_r in range(n_runs):
        # Optimizing Poses
        if i_r % n_runs_per_run_pose == 0:
            g, pos, poses_test_length, poses_training_length = optimize_poses(
                g, pos, map_img, optimizer_pos, rng)
            if i_r % stats_and_eval_every == 0:
                stats.add("poses_test_length", i_r, float(poses_test_length))
                stats.add("poses_training_length", i_r,
                          float(poses_training_length))

        # Optimizing Policy
        if i_r % n_runs_per_run_policy == 0:
            (policy_model, env_nx, policy_loss, new_data_percentage,
             policy_data_files, data_len
             ) = optimize_policy(
                policy_model, g, n_agents, n_epochs_per_run_policy,
                batch_size_policy, optimizer_policy, policy_data_files,
                pool, prefix, rng)
            if i_r % stats_and_eval_every == 0:
                # also eval now
                rng_test = Random(1)
                n_eval = 10
                # little less agents for evaluation
                eval_n_agents = int(np.ceil(n_agents * .7))
                regret, success = eval_policy_full_scenario(
                    policy_model, g, env_nx, eval_n_agents, n_eval, rng_test)
                policy_accuracy = policy_model.accuracy(policy_eval_list)

                stats.add("policy_loss", i_r, float(policy_loss))
                if regret is not None:
                    stats.add("policy_regret", i_r, float(regret))
                stats.add("policy_success", i_r, float(success))
                stats.add("policy_accuracy", i_r, policy_accuracy)
                stats.add("policy_new_data_percentage",
                          i_r, float(new_data_percentage))
                stats.add("n_policy_data_len", i_r, float(data_len))
                logger.info(f"Regret: {regret}")
                logger.info(f"Success: {success}")
                logger.info(f"Accuracy: {policy_accuracy:.2f}")
                logger.info(f"New data: {new_data_percentage*100:.1f}%")
                logger.info(f"Data length: {data_len}")

        pb.progress()
    runtime = pb.end()
    stats.add_static("runtime", str(runtime))

    draw_graph(g, map_img, title="End")
    plt.savefig(f"multi_optim/results/{prefix}_end.png")

    fig, axs = plt.subplots(2, 1, sharex=True)
    for i_x, part in enumerate(["poses", "policy"]):
        for k, v in stats.get_stats_wildcard(f"{part}.*").items():
            axs[i_x].plot(v[0], v[1], label=k)
        axs[i_x].legend()
        axs[i_x].xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel("Run")
    plt.savefig(f"multi_optim/results/{prefix}_stats.png")

    # Save results
    stats.to_yaml(f"multi_optim/results/{prefix}_stats.yaml")
    nx.write_gpickle(g, f"multi_optim/results/{prefix}_graph.gpickle")
    torch.save(policy_model.state_dict(),
               f"multi_optim/results/{prefix}_policy_model.pt")

    logger.info(stats.get_statics())

    return g, pos, poses_test_length, poses_training_length


if __name__ == "__main__":
    tracemalloc.start()

    # multiprocessing
    tmp.set_sharing_strategy('file_system')
    tmp.set_start_method('spawn')

    # debug run
    rng = Random(0)
    logging.getLogger(__name__).setLevel(logging.DEBUG)
    logging.getLogger(
        "planner.mapf_implementations.plan_cbs_roadmap"
    ).setLevel(logging.DEBUG)
    # logging.getLogger(
    #     "sim.decentralized.policy"
    # ).setLevel(logging.DEBUG)
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
        rng=rng,
        prefix="tiny")

    # # checking different metaparams ...
    # diffs = {
    #     "n_agents": [4, 5],
    #     "lr_policy": [3e-3, 3e-4],
    #     # "n_epochs_per_run_policy": [32, 64],
    # }  # type: Dict[str, List[float]]
    # def_n_agents = 6
    # def_lr_policy = 1e-3
    # def_n_epochs_per_run_policy = 128
    # for k, vs in diffs.items():
    #     for v in vs:
    #         rng = Random(0)
    #         args = {
    #             "n_nodes": 16,
    #             "n_runs_pose": 2,
    #             "n_runs_policy": 128,
    #             "stats_and_eval_every": 2,
    #             "lr_pos": 1e-4,
    #             "lr_policy": def_lr_policy,
    #             "n_agents": def_n_agents,
    #             "map_fname": "roadmaps/odrm/odrm_eval/maps/x.png",
    #             "rng": rng,
    #             "prefix": f"tiny_{k}_{v}",
    #         }
    #         args[k] = v
    #         run_optimization(**args)  # type: ignore
