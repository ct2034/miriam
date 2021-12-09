import random
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import png
import torch
from matplotlib import pyplot as plt
from roadmaps.var_odrm_torch.var_odrm_torch import (draw_graph, make_graph,
                                                    optimize_poses, read_map,
                                                    sample_points)
from sim.decentralized.agent import Agent
from sim.decentralized.iterators import IteratorType
from sim.decentralized.policy import Policy, PolicyType
from sim.decentralized.runner import run_a_scenario
from tools import ProgressBar


def find_collisions(agents: List[Agent]
                    ) -> Dict[Tuple[int, int], Tuple[int, int]]:
    agent_visited: Dict[Tuple[int, int], int] = {}  # {(node, t): agent}
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


def optimize_policy(g: nx.Graph, pos, n_agents, rng):
    n_nodes = g.number_of_nodes()
    has_collisions = False
    fixed_env_nx: Optional[nx.Graph] = None
    while not has_collisions:
        agents: List[Agent] = []
        starts_goals = rng.choice(
            np.arange(n_nodes), size=(n_agents, 2), replace=False)
        for i_ga in range(n_agents):
            start, goal = starts_goals[i_ga, :]
            policy = PolicyType.RANDOM
            if fixed_env_nx is None:
                agent = Agent(pos, (start,), policy)
                fixed_env_nx = agent.env_nx.copy()
            else:
                agent = Agent(pos, (start,), policy,
                              env_nx=fixed_env_nx.copy())
            agent.give_a_goal((goal,))
            agents.append(agent)
        collisions = find_collisions(agents)
        has_collisions = (len(collisions) > 0)
    collision_nodes = np.array(list(collisions.keys()))
    first = np.argmin(collision_nodes[:, 1])
    col_node, col_t = collision_nodes[first]
    col_agents = collisions[(col_node, col_t)]
    print(f"col_node: {col_node}, col_t: {col_t}, col_agents: {col_agents}")
    # save results when this agent has prio
    results: Dict[int, Tuple[float, float, float, float, float]] = {}
    for i_ca, i_a in enumerate(col_agents):
        i_oa = col_agents[i_ca % len(col_agents)]
        agents = []
        for i_ga in range(n_agents):
            start, goal = starts_goals[i_ga, :]
            if i_ga == i_a:
                policy = PolicyType.ONE_THEN_RANDOM
            elif i_ga == i_oa:
                policy = PolicyType.ZERO_THEN_RANDOM
            else:
                policy = PolicyType.RANDOM
            assert fixed_env_nx is not None
            agent = Agent(pos, (start,), policy,
                          env_nx=fixed_env_nx.copy())
            agent.give_a_goal((goal,))
            agents.append(agent)
        results[i_a] = run_a_scenario(
            None, agents, False, IteratorType.BLOCKING1)
    for i_a, r in results.items():
        print(f"Agent {i_a}")
        (average_time, max_time, average_length,  max_length, successful
         ) = r
        print(f" average_time: {average_time}, max_time: {max_time},\n" +
              f" average_length: {average_length}," +
              f" max_length: {max_length},\n" +
              f" successful: {successful}")


def run_optimization(
        n_nodes: int = 64,
        n_runs: int = 16,
        lr_pos: float = 1e-4,
        n_agents: int = 4,
        map_fname: str = "roadmaps/odrm_eval/maps/z.png"):
    map_img = read_map(map_fname)
    pos = sample_points(n_nodes, map_img)
    optimizer_pos = torch.optim.Adam([pos], lr=lr_pos)
    g = make_graph(pos, map_img)

    draw_graph(g, pos)
    plt.show()

    pb = ProgressBar("Optimization", n_runs)
    for i_r in range(n_runs):
        # Optimizing Poses
        g, pos, test_length, training_length = optimize_poses(
            g, pos, map_img, optimizer_pos)

        # Optimizing Agents
        rng = np.random.RandomState(i_r)
        optimize_policy(g, pos, n_agents, rng)
        pb.progress()

    pb.end()

    draw_graph(g, pos)
    plt.show()


if __name__ == "__main__":
    run_optimization()
