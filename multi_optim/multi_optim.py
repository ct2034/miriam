import random
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


def find_collisions(agents: List[Agent]
                    ) -> Dict[Tuple[int, int], Tuple[int, int]]:
    agent_visited: Dict[Tuple[int, int], int] = {}  # {(node, t): agent}
    # {(node, t): (agent1, agent2)}
    collisions: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for i_a, a in enumerate(agents):
        assert a.path is not None
        for t, node in enumerate(a.path):
            timed_node = (node, t)
            if timed_node in agent_visited.keys():
                collisions[timed_node] = (
                    agent_visited[timed_node],
                    i_a
                )
            agent_visited[timed_node] = i_a
    return collisions


def optimize_policy(g: nx.Graph, pos, n_agents, rng):
    n_nodes = g.number_of_nodes()
    has_collisions = False
    while not has_collisions:
        agents = []
        starts_goals = rng.choice(
            np.arange(n_nodes), size=(n_agents, 2), replace=False)
        for i_a in range(n_agents):
            start, goal = starts_goals[i_a, :]
            if len(agents) == 0:
                a = Agent(pos, (start,))
            else:
                a = Agent(pos, (start,), env_nx=agents[0].env_nx.copy())
            a.give_a_goal((goal,))
            agents.append(a)
        collisions = find_collisions(agents)
        has_collisions = (len(collisions) > 0)
    (average_time, max_time, average_length,  max_length, successful
     ) = run_a_scenario(None, agents, False, IteratorType.WAITING)
    print(f"average_time{average_time}, max_time{max_time}, " +
          f"average_length{average_length}, max_length{max_length}, " +
          f"successful{successful}")


def run_optimization(
        n_nodes: int = 100,
        n_runs: int = 16,
        lr_pos: float = 1e-4,
        n_agents: int = 8):
    pos = sample_points(n_nodes)
    optimizer_pos = torch.optim.Adam([pos], lr=lr_pos)
    g = make_graph(pos)

    pb = ProgressBar("Optimization", n_runs)
    for i_r in range(n_runs):
        # Optimizing Poses
        g, pos, test_length, training_length = optimize_poses(
            g, pos, optimizer_pos)

        # Optimizing Agents
        rng = np.random.RandomState(i_r)
        optimize_policy(g, pos, n_agents, rng)
        pb.progress()

    pb.end()


if __name__ == "__main__":
    run_optimization()
