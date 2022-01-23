import datetime
import logging
import socket
import sys
from fileinput import filename
from random import Random, getstate
from typing import Dict, List, Optional, Tuple

import git
import networkx as nx
import numpy as np
import torch
import torch.multiprocessing as tmp
import torch_geometric
from definitions import INVALID, SCENARIO_RESULT
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from planner.policylearn.edge_policy import EdgePolicyModel
from planner.policylearn.edge_policy_graph_utils import RADIUS
from roadmaps.var_odrm_torch.var_odrm_torch import (draw_graph, make_graph,
                                                    optimize_poses, read_map,
                                                    sample_points)
from sim.decentralized.agent import Agent
from sim.decentralized.iterators import IteratorType
from sim.decentralized.policy import EdgePolicy, OptimalEdgePolicy
from sim.decentralized.runner import run_a_scenario
from tools import ProgressBar, StatCollector

import dagger

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


def eval_policy(model, g: nx.Graph, env_nx: nx.Graph, n_agents, n_eval, rng
                ) -> Tuple[Optional[float], float]:
    regret_s = []
    success_s = []
    for i_e in range(n_eval):
        logger.debug(f"Eval {i_e}")
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
                        iterator=IteratorType.EDGE_POLICY3)
                elif policy is OptimalEdgePolicy:
                    try:
                        res_optim = run_a_scenario(
                            env=g,
                            agents=tuple(agents),
                            plot=False,
                            iterator=IteratorType.EDGE_POLICY3)
                    except Exception as e:
                        logger.error(e)
                        res_optim = (0, 0, 0, 0, 0)

        success = res_policy[4] and res_optim[4]
        logger.debug(f"success: {success}")
        if success:
            regret = res_policy[0] - res_optim[0]
            regret_s.append(regret)
            logger.debug(f"regret: {regret}")
        success_s.append(res_policy[4])
    if len(regret_s) > 0:
        return np.mean(regret_s), np.mean(success_s)
    else:
        return None, np.mean(success_s)


def optimize_policy(model, g: nx.Graph, n_agents, optimizer, old_ds, pool, rng):
    n_epochs = 64
    ds = dagger.DaggerStrategy(
        model, g, n_epochs, n_agents, optimizer, rng)
    model, loss, old_ds = ds.run_dagger(pool, old_ds)

    rng_test = Random(1)
    # little less agents for evaluation
    eval_n_agents = int(np.ceil(n_agents * .7))
    regret, success = eval_policy(
        model, g, ds.env_nx, eval_n_agents, 10, rng_test)

    return model, loss, regret, success, old_ds


def run_optimization(
        n_nodes: int = 32,
        n_runs_pose: int = 1024,
        n_runs_policy: int = 128,
        stats_every: int = 1,
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
        device = torch.device("cuda:0")
    else:
        logger.warning("GPU not available, using CPU")
        device = torch.device("cpu")

    # Policy
    policy_model = EdgePolicyModel()
    policy_model.to(device)
    # policy_model.use_multiprocessing = False
    # policy_model.share_memory()
    optimizer_policy = torch.optim.Adam(
        policy_model.parameters(), lr=lr_policy)
    old_ds = []

    # Visualization and analysis
    stats = StatCollector([
        "poses_test_length",
        "poses_training_length",
        "policy_loss",
        "policy_regret",
        "policy_success"])
    stats.add_statics({
        # metadata
        "hostname": socket.gethostname(),
        "git_hash": git.Repo(".").head.object.hexsha,
        "started_at": datetime.datetime.now().isoformat(),
        # parameters
        "n_nodes": n_nodes,
        "n_runs_pose": n_runs_pose,
        "n_runs_policy": n_runs_policy,
        "stats_every": stats_every,
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
            if i_r % stats_every == 0:
                stats.add("poses_test_length", i_r, float(poses_test_length))
                stats.add("poses_training_length", i_r,
                          float(poses_training_length))

        # Optimizing Policy
        if i_r % n_runs_per_run_policy == 0:
            (policy_model, policy_loss, regret, success, old_ds
             ) = optimize_policy(
                policy_model, g, n_agents, optimizer_policy, old_ds, pool, rng)
            if i_r % stats_every == 0:
                stats.add("policy_loss", i_r, float(policy_loss))
                if regret is not None:
                    stats.add("policy_regret", i_r, float(regret))
                stats.add("policy_success", i_r, float(success))
                logger.info(f"Regret: {regret}")
                logger.info(f"Success: {success}")

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
    logging.getLogger(__name__).setLevel(logging.DEBUG)

    # multiprocessing
    tmp.set_sharing_strategy('file_system')
    tmp.set_start_method('fork')

    # debug run
    rng = Random(0)
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
        stats_every=1,
        lr_pos=1e-4,
        lr_policy=1e-3,
        n_agents=4,
        map_fname="roadmaps/odrm/odrm_eval/maps/x.png",
        rng=rng,
        prefix="debug")

    # small run
    rng = Random(0)
    logging.getLogger(
        "planner.mapf_implementations.plan_cbs_roadmap"
    ).setLevel(logging.INFO)
    logging.getLogger(
        "sim.decentralized.policy"
    ).setLevel(logging.INFO)
    run_optimization(
        n_nodes=16,
        n_runs_pose=1,
        n_runs_policy=128,
        stats_every=1,
        lr_pos=1e-4,
        lr_policy=1e-4,
        n_agents=6,
        map_fname="roadmaps/odrm/odrm_eval/maps/plain.png",
        rng=rng,
        prefix="small")

    # full run
    rng = Random(0)
    logging.getLogger(
        "planner.mapf_implementations.plan_cbs_roadmap"
    ).setLevel(logging.INFO)
    run_optimization(
        rng=rng,
        prefix="full")
