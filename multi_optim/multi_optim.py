import logging
from random import Random
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
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
                        logging.error(e)
                        res_optim = (0, 0, 0, 0, 0)

        success = res_policy[4] and res_optim[4]
        if success:
            if res_policy[0] == 0:
                regret = 0.
            else:
                # regret as percentage of suboptimal path
                regret = (res_policy[0] - res_optim[0]) / res_policy[0]
            regret_s.append(regret)
        success_s.append(res_policy[4])
    if len(regret_s) > 0:
        return np.mean(regret_s), np.mean(success_s)
    else:
        return None, np.mean(success_s)


def optimize_policy(model, g: nx.Graph, n_agents, optimizer, old_d, rng):
    n_epochs = 8
    ds = dagger.DaggerStrategy(
        model, g, n_epochs, n_agents, optimizer, old_d, rng)
    model, loss = ds.run_dagger()

    rng_test = Random(1)
    # little less agents for evaluation
    eval_n_agents = int(np.ceil(n_agents * .7))
    regret, success = eval_policy(
        model, g, ds.env_nx, eval_n_agents, 10, rng_test)

    return model, loss, regret, success, ds.d


def run_optimization(
        n_nodes: int = 32,
        n_runs_pose: int = 1024,
        n_runs_policy: int = 128,
        stats_every: int = 1,
        lr_pos: float = 1e-4,
        lr_policy: float = 1e-4,
        n_agents: int = 8,
        map_fname: str = "roadmaps/odrm/odrm_eval/maps/x.png",
        rng: Random = Random(0),
        prefix: str = "noname"):
    # Roadmap
    map_img = read_map(map_fname)
    pos = sample_points(n_nodes, map_img, rng)
    optimizer_pos = torch.optim.Adam([pos], lr=lr_pos)
    g = make_graph(pos, map_img)

    # Policy
    policy_model = EdgePolicyModel(rng=rng)
    optimizer_policy = torch.optim.Adam(
        policy_model.parameters(), lr=lr_policy)
    old_d = None

    # Visualization and analysis

    stats = StatCollector([
        "poses_test_length",
        "poses_training_length",
        "policy_loss",
        "policy_regret",
        "policy_success"])
    draw_graph(g, map_img, title="Start")
    plt.savefig(f"multi_optim/results/{prefix}_start.png")

    # Making sense of two n_runs
    assert n_runs_pose > n_runs_policy
    n_runs = n_runs_pose
    n_runs_pose_per_policy = n_runs // n_runs_policy

    # Run optimization
    pb = ProgressBar(f"{prefix} Optimization", n_runs, 1)
    for i_r in range(n_runs):
        # Optimizing Poses
        g, pos, poses_test_length, poses_training_length = optimize_poses(
            g, pos, map_img, optimizer_pos, rng)
        if i_r % stats_every == 0:
            stats.add("poses_test_length", i_r, float(poses_test_length))
            stats.add("poses_training_length", i_r,
                      float(poses_training_length))

        if i_r % n_runs_pose_per_policy == 0:
            # Optimizing Policy
            (policy_model, policy_loss, regret, success, old_d
             ) = optimize_policy(
                policy_model, g, n_agents, optimizer_policy, old_d, rng)
            stats.add("policy_loss", i_r, float(policy_loss))
            if regret is not None:
                stats.add("policy_regret", i_r, float(regret))
            stats.add("policy_success", i_r, float(success))

        pb.progress()
    pb.end()

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

    return g, pos, poses_test_length, poses_training_length


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # debug run
    rng = Random(0)
    logging.getLogger(
        "planner.mapf_implementations.plan_cbs_roadmap"
    ).setLevel(logging.DEBUG)
    run_optimization(
        n_nodes=8,
        n_runs_pose=4,
        n_runs_policy=2,
        stats_every=1,
        lr_pos=1e-4,
        lr_policy=1e-4,
        n_agents=2,
        map_fname="roadmaps/odrm/odrm_eval/maps/x.png",
        rng=rng,
        prefix="debug")

    # small run
    rng = Random(0)
    logging.getLogger(
        "planner.mapf_implementations.plan_cbs_roadmap"
    ).setLevel(logging.INFO)
    run_optimization(
        n_nodes=16,
        n_runs_pose=512,
        n_runs_policy=256,
        stats_every=1,
        lr_pos=1e-4,
        lr_policy=1e-4,
        n_agents=8,
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
