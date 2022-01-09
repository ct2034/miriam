import logging
from random import Random
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
from definitions import INVALID, SCENARIO_RESULT
from matplotlib import pyplot as plt
from planner.policylearn.edge_policy import EdgePolicyModel
from planner.policylearn.edge_policy_graph_utils import RADIUS
from roadmaps.var_odrm_torch.var_odrm_torch import (draw_graph, make_graph,
                                                    optimize_poses, read_map,
                                                    sample_points)
from sim.decentralized.agent import Agent
from sim.decentralized.iterators import IteratorType
from sim.decentralized.policy import EdgePolicy, OptimalEdgePolicy
from sim.decentralized.runner import run_a_scenario
from tools import ProgressBar

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
        for policy in [OptimalEdgePolicy, EdgePolicy]:
            agents = []
            for i_a in range(n_agents):
                a = Agent(g, starts[i_a], env_nx=env_nx, radius=RADIUS)
                a.give_a_goal(goals[i_a])
                a.policy = policy(a, model)
                agents.append(a)
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
                except RuntimeError:
                    res_optim = (0, 0, 0, 0, 0)

        success = res_policy[4] and res_optim[4]
        if success:
            regret_s.append(res_policy[0] - res_optim[0])
        success_s.append(res_policy[4])
    if len(regret_s) > 0:
        return np.mean(regret_s), np.mean(success_s)
    else:
        return None, np.mean(success_s)


def optimize_policy(model, g: nx.Graph, n_agents, optimizer, old_d, rng):
    ds = dagger.DaggerStrategy(model, g, 2, n_agents, optimizer, old_d, rng)
    model, loss = ds.run_dagger()

    rng_test = Random(1)
    regret, success = eval_policy(model, g, ds.env_nx, n_agents, 10, rng_test)

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
        rng: Random = Random(0)):
    # Roadmap
    map_img = read_map(map_fname)
    pos = sample_points(n_nodes, map_img, rng)
    optimizer_pos = torch.optim.Adam([pos], lr=lr_pos)
    g = make_graph(pos, map_img)

    # Policy
    policy_model = EdgePolicyModel()
    optimizer_policy = torch.optim.Adam(
        policy_model.parameters(), lr=lr_policy)
    old_d = None

    # Visualization and analysis
    stats: Dict[str, Dict[str, List[float]]] = {
        "poses_test_length": {
            "x": [],
            "t": []
        },
        "poses_training_length": {
            "x": [],
            "t": []
        },
        "policy_loss": {
            "x": [],
            "t": []
        },
        "policy_regret": {
            "x": [],
            "t": []
        },
        "policy_success": {
            "x": [],
            "t": []
        }
    }
    draw_graph(g, map_img, title="Start")
    plt.savefig("multi_optim/start.png")

    # Making sense of two n_runs
    assert n_runs_pose > n_runs_policy
    n_runs = n_runs_pose
    n_runs_pose_per_policy = n_runs // n_runs_policy

    # Run optimization
    pb = ProgressBar("Optimization", n_runs, 1)
    for i_r in range(n_runs):
        # Optimizing Poses
        g, pos, poses_test_length, poses_training_length = optimize_poses(
            g, pos, map_img, optimizer_pos, rng)
        if i_r % stats_every == 0:
            stats["poses_test_length"]["x"].append(poses_test_length)
            stats["poses_test_length"]["t"].append(i_r)
            stats["poses_training_length"]["x"].append(poses_training_length)
            stats["poses_training_length"]["t"].append(i_r)

        if i_r % n_runs_pose_per_policy == 0:
            # Optimizing Policy
            (policy_model, policy_loss, regret, success, old_d
             ) = optimize_policy(
                policy_model, g, n_agents, optimizer_policy, old_d, rng)
            stats["policy_loss"]["x"].append(policy_loss)
            stats["policy_loss"]["t"].append(i_r)
            if regret is not None:
                stats["policy_regret"]["x"].append(regret)
                stats["policy_regret"]["t"].append(i_r)
            stats["policy_success"]["x"].append(success)
            stats["policy_success"]["t"].append(i_r)

        pb.progress()
    pb.end()

    draw_graph(g, map_img, title="End")
    plt.savefig("multi_optim/end.png")

    plt.figure()
    for k, v in stats.items():
        plt.plot(v["t"], v["x"], label=k)
    plt.xlabel("Run")
    plt.legend()
    plt.savefig("multi_optim/stats.png")

    return g, pos, poses_test_length, poses_training_length


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # logging.getLogger(
    #     "planner.mapf_implementations.plan_cbs_roadmap"
    # ).setLevel(logging.DEBUG)

    rng = Random(0)
    run_optimization(rng=rng)
