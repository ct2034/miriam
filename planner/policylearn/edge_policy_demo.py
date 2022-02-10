import logging
from itertools import product
from random import Random
from typing import Any, List

import networkx as nx
import scenarios
import sim
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
from torch_geometric.data import Data


def make_random_data_self(rng, n_nodes, num_node_features):
    fully_connected_edge_index = torch.tensor(
        [[a, b] for a, b in product(range(n_nodes), repeat=2)],
        dtype=torch.long
    ).T
    node = rng.choice(range(n_nodes))
    x = torch.rand(n_nodes, num_node_features) * .5
    x[node, 0] = 1.
    return Data(
        x=x,
        edge_index=fully_connected_edge_index,
        y=node
    )


def demo_learning():
    logging.getLogger('sim.decentralized.policy').setLevel(logging.DEBUG)
    logging.getLogger('sim.decentralized.agent').setLevel(logging.DEBUG)
    logging.getLogger('sim.decentralized.runner').setLevel(logging.DEBUG)
    logging.getLogger('sim.decentralized.iterators').setLevel(logging.DEBUG)

    seed = 0
    rng = Random(seed)
    torch.manual_seed(seed)
    n_nodes = 3
    big_from_small = {n: n for n in range(n_nodes)}
    num_node_features = 2
    conv_channels = 4
    model = EdgePolicyModel(num_node_features, conv_channels)

    # eval set
    eval_set = [make_random_data_self(rng, n_nodes, num_node_features)
                for _ in range(50)]

    # learning to always use self edge
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    for i in range(int(1E3)):
        ds = [make_random_data_self(rng, n_nodes, num_node_features)
              for _ in range(10)]
        loss = model.learn(ds, optimizer)
        if i % 1E2 == 0:
            print(f"loss: {loss:.3f}")
            accuracy = model.accuracy(
                eval_set, [big_from_small for _ in range(len(eval_set))])
            print(f"accuracy: {accuracy:.3f}")

    # trying in a scenario
    # rng = Random(0)
    # (env, starts, goals) = arena_with_crossing(4, 0, 6, rng)
    # env_g = gridmap_to_nx(env)
    # starts_g = starts_or_goals_to_nodes(starts, env)
    # goals_g = starts_or_goals_to_nodes(goals, env)
    # agents = scenarios.evaluators.to_agent_objects(env_g, starts_g, goals_g,
    #                                                policy=PolicyType.OPTIMAL_EDGE,  # will be overwritten below
    #                                                radius=.1)
    # for a in agents:
    #     a.policy = sim.decentralized.policy.EdgePolicy(a, model)

    # paths: List[Any] = []
    # stats = run_a_scenario(env, agents, plot=False,
    #                        iterator=IteratorType.EDGE_POLICY3,
    #                        paths_out=paths)
    # print(stats)
    # plot_with_paths(env_g, paths)
    # plt.show()


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
