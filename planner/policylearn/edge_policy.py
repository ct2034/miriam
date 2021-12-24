import logging
from random import Random

import matplotlib.pyplot as plt
import sim.decentralized.policy
import torch
import torch.nn as nn
import torch_geometric
from scenarios.evaluators import to_agent_objects
from scenarios.generators import arena_with_crossing
from scenarios.graph_converter import gridmap_to_nx, starts_or_goals_to_nodes
from scenarios.visualization import plot_with_paths
from sim.decentralized.iterators import IteratorType
from sim.decentralized.runner import run_a_scenario


class EdgePolicyModel(nn.Module):
    def __init__(self, num_node_features, conv_channels):
        super().__init__()
        self.conv1 = torch_geometric.nn.GCNConv(
            num_node_features, conv_channels)
        self.conv2 = torch_geometric.nn.GCNConv(
            conv_channels, conv_channels)
        self.readout = torch.nn.Linear(conv_channels, 1)

    def forward(self, x, edge_index, pos, node):
        # Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()

        # Relate to edges
        # take only edges that start or end at node
        relevant_edge_index = edge_index[:,
                                         torch.bitwise_or(edge_index[0] == node,
                                                          edge_index[1] == node)]
        # make index of targets
        targets = torch.zeros(relevant_edge_index.shape[1], dtype=torch.long)
        for i, (s, d) in enumerate(relevant_edge_index.t()):
            if s == node and d == node:
                targets[i] = node
            elif s == node and d != node:
                targets[i] = d
            elif s != node and d == node:
                targets[i] = s
            else:
                raise ValueError("Edge not found")

        score = self.readout(x[targets])
        score = torch.softmax(score, dim=0)
        return score[:, 0], targets


if __name__ == "__main__":
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
    score, targets = model(x, edge_index, pos, node)

    # learning to always use self edge
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for i in range(100):
        model.train()
        node = 0
        score, targets = model(x, edge_index, pos, node)
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
    (env, starts, goals) = arena_with_crossing(8, 0, 8, rng)
    env_g = gridmap_to_nx(env)
    starts_g = starts_or_goals_to_nodes(starts, env)
    goals_g = starts_or_goals_to_nodes(goals, env)
    agents = to_agent_objects(env_g, starts_g, goals_g)

    for a in agents:
        a.policy = sim.decentralized.policy.EdgePolicy(a, model)
    paths = []
    stats = run_a_scenario(env, agents, plot=False,
                           iterator=IteratorType.EDGE_POLICY3,
                           paths_out=paths)
    print(stats)
    plot_with_paths(env_g, paths)
    plt.show()
