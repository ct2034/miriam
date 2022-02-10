import logging
from random import Random
from typing import Any, List

import networkx as nx
import scenarios
import torch
from definitions import INVALID, POS
from matplotlib import pyplot as plt
from planner.policylearn.edge_policy import EdgePolicyModel
from planner.policylearn.edge_policy_graph_utils import agents_to_data
from planner.policylearn.generate_data_demo import plot_graph_wo_pos_data
from scenarios.generators import arena_with_crossing
from scenarios.graph_converter import gridmap_to_nx, starts_or_goals_to_nodes
from scenarios.visualization import plot_with_paths
from sim.decentralized.agent import Agent
from sim.decentralized.iterators import IteratorType
from sim.decentralized.policy import PolicyType
from sim.decentralized.runner import run_a_scenario


def demo_learning():
    logging.getLogger('sim.decentralized.policy').setLevel(logging.DEBUG)
    logging.getLogger('sim.decentralized.agent').setLevel(logging.DEBUG)
    logging.getLogger('sim.decentralized.runner').setLevel(logging.DEBUG)
    logging.getLogger('sim.decentralized.iterators').setLevel(logging.DEBUG)

    n_nodes = 6
    num_node_features = 2
    conv_channels = 4
    model = EdgePolicyModel(num_node_features, conv_channels)
    x = torch.randn(n_nodes, num_node_features)
    edge_index = torch.tensor([
        [0, 0, 0, 0, 1, 0, 3, 5],
        [1, 2, 3, 4, 4, 0, 1, 2]
    ])
    pos = torch.tensor([
        [0, 0],
        [1, 0],
        [0, 1],
        [-1, 0],
        [0, -1],
        [1, 1]
    ])
    node = 0
    x[node, 0] = 1.
    score, targets = model(x, edge_index)

    # learning to always use self edge
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for i in range(100):
        model.train()
        node = 0
        x[node, 0] = 1.
        score, targets = model(x, edge_index)
        score_optimal = torch.zeros(score.shape)
        score_optimal[targets == node] = 1
        # bce loss
        loss = torch.nn.functional.binary_cross_entropy(
            score, score_optimal)
        if i % 10 == 0:
            print(" ".join([f"{s:.3f}" for s in score.tolist()]))
            print(f"loss: {loss.item():.3f}")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # trying in a scenario
    rng = Random(0)
    (env, starts, goals) = arena_with_crossing(4, 0, 6, rng)
    env_g = gridmap_to_nx(env)
    starts_g = starts_or_goals_to_nodes(starts, env)
    goals_g = starts_or_goals_to_nodes(goals, env)
    agents = scenarios.evaluators.to_agent_objects(env_g, starts_g, goals_g,
                                                   policy=PolicyType.OPTIMAL_EDGE)

    # for a in agents:
    #     a.policy = sim.decentralized.policy.EdgePolicy(a, model)

    paths: List[Any] = []
    stats = run_a_scenario(env, agents, plot=False,
                           iterator=IteratorType.EDGE_POLICY3,
                           paths_out=paths)
    print(stats)
    plot_with_paths(env_g, paths)
    plt.show()


def demo_graph():
    g = nx.Graph()
    g.add_edges_from([
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 0),
        (1, 5),
        (3, 7),
        (5, 6),
        (6, 7),
        (3, 8),
        (7, 8),
    ])
    nx.set_node_attributes(g, {
        0: (0., 0.),
        1: (1., 0.),
        2: (1., 1.),
        3: (.5, 1.7),
        4: (0., 1.),
        5: (2., 0.),
        6: (2., 1.),
        7: (1.5, 1.7),
        8: (1, 3.4),
    }, POS)

    agents = [
        Agent(g, 0, radius=.1),
        Agent(g, 3, radius=.1)
    ]
    agents[0].give_a_goal(5)
    agents[1].give_a_goal(0)

    # each make first step
    # agents[0].make_next_step(agents[0].path[1][0])
    # agents[1].make_next_step(agents[1].path[1][0])

    data, _, big_from_small = agents_to_data(
        agents=agents,
        i_self=0,
        hop_dist=2)
    print(data.x)
    pos = {
        i_n: g.nodes[big_from_small[i_n]][POS] for i_n in range(data.num_nodes)
    }

    f, ax = plt.subplots(1, 1)
    plot_graph_wo_pos_data(ax, data.edge_index, pos, data.x)
    plt.show()


if __name__ == "__main__":
    demo_learning()
