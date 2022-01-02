import logging
from random import Random
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
from matplotlib import pyplot as plt
from roadmaps.var_odrm_torch.var_odrm_torch import (draw_graph, make_graph,
                                                    optimize_poses, read_map,
                                                    sample_points)
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


def optimize_policy(model, g: nx.Graph, n_agents, rng):
    ds = dagger.DaggerStrategy(model, g, 2, n_agents, rng)
    model, loss = ds.run_dagger()
    return model, loss


def run_optimization(
        n_nodes: int = 16,
        n_runs_pose: int = 1024,
        n_runs_policy: int = 8,
        stats_every: int = 1,
        lr_pos: float = 1e-4,
        n_agents: int = 3,
        map_fname: str = "roadmaps/odrm/odrm_eval/maps/x.png",
        rng: Random = Random(0)):
    # Roadmap
    map_img = read_map(map_fname)
    pos = sample_points(n_nodes, map_img, rng)
    optimizer_pos = torch.optim.Adam([pos], lr=lr_pos)
    g = make_graph(pos, map_img)

    # Policy
    policy_model = None  # start with no policy

    # Visualization and analysis
    stats = {
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

        if i_r % n_runs_pose_per_policy == 0:
            # Optimizing Policy
            policy_model, policy_loss = optimize_policy(
                policy_model, g, n_agents, rng)

        # Saving stats
        if i_r % stats_every == 0:
            stats["poses_test_length"]["x"].append(poses_test_length)
            stats["poses_test_length"]["t"].append(i_r)
            stats["poses_training_length"]["x"].append(poses_training_length)
            stats["poses_training_length"]["t"].append(i_r)
            stats["policy_loss"]["x"].append(policy_loss)
            stats["policy_loss"]["t"].append(i_r)
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
    logging.getLogger(
        "planner.mapf_implementations.plan_cbs_roadmap").setLevel(logging.DEBUG)

    rng = Random(0)
    run_optimization(rng=rng)
