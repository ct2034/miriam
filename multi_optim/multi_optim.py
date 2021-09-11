from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import torch
from roadmaps.var_odrm_torch.var_odrm_torch import (make_graph, optimize_poses,
                                                    sample_points)
from sim.decentralized.agent import Agent
from sim.decentralized.iterators import IteratorType
from sim.decentralized.runner import run_a_scenario
from tools import ProgressBar


def optimize_policy(g: nx.Graph, pos, n_agents):
    n_nodes = g.number_of_nodes()
    agents = []
    starts_goals = np.random.choice(
        np.arange(n_nodes), size=(n_agents, 2), replace=False)
    for i_a in range(n_agents):
        start, goal = starts_goals[i_a, :]
        a = Agent(pos, int(start), env_nx=g)
        a.give_a_goal(int(goal))
        agents.append(a)
    run_a_scenario(None, agents, False, IteratorType.WAITING)


def run_optimization(
        n_nodes: int = 100,
        n_runs: int = 16,
        lr_pos: float = 1e-4,
        n_agents: int = 4):
    pos = sample_points(n_nodes)
    optimizer_pos = torch.optim.Adam([pos], lr=lr_pos)
    g = make_graph(pos)

    pb = ProgressBar("Optimization", n_runs)
    for i_r in range(n_runs):
        # Optimizing Poses
        g, pos, test_length, training_length = optimize_poses(
            g, pos, optimizer_pos)

        # Optimizing Agents
        optimize_policy(g, pos, n_agents)
        pb.progress()

    pb.end()


if __name__ == "__main__":
    run_optimization()
