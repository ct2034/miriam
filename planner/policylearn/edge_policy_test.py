import unittest
from cgi import test
from itertools import product
from random import Random
from timeit import repeat

import torch
from planner.policylearn.edge_policy import EdgePolicyModel
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch


def make_data(rng, num_node_features, n_nodes):
    node = rng.randint(0, n_nodes - 1)
    data_x = torch.tensor(
        [[rng.random() for _ in range(num_node_features)] for _ in range(n_nodes)]
    )
    data_x[node, 0] = 1.0  # learn from this
    data_edge_index = torch.tensor(
        [[a, b] for a, b in product(range(n_nodes), repeat=2)]
    ).T
    data_y = torch.zeros(n_nodes, dtype=torch.float32)
    data_y[node] = 1.0
    data = Data(x=data_x, edge_index=data_edge_index, y=data_y)
    return data


class EdgePolicyTest(unittest.TestCase):
    def test_edge_policy(self):
        n_nodes = 5
        num_node_features = 2
        conv_channels = 2
        policy = EdgePolicyModel(num_node_features, conv_channels)
        x = torch.randn(n_nodes, num_node_features)
        x[0, 0] = 1  # node 1 is our node
        edge_index = torch.tensor([[0, 0, 0, 0], [1, 2, 3, 4]])
        score = policy(x, edge_index, torch.tensor([0] * n_nodes))
        self.assertEqual(score.shape, (n_nodes,))

    def test_edge_policy_learn(self):
        rng = Random(0)
        torch.manual_seed(0)
        num_node_features = 2
        conv_channels = 3
        n_nodes = 2
        n_epochs = 20
        n_data = 10
        policy = EdgePolicyModel(num_node_features, conv_channels)
        optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
        datas = []

        # test data
        test_data = [
            (make_data(rng, num_node_features, n_nodes), {x: x for x in range(n_nodes)})
            for _ in range(10)
        ]
        test_acc = policy.accuracy(test_data)
        self.assertLess(test_acc, 0.6)

        # train
        for _ in range(n_epochs):
            for _ in range(n_data):
                datas.append(make_data(rng, num_node_features, n_nodes))
            policy.learn(Batch.from_data_list(datas), optimizer)
            test_acc = policy.accuracy(test_data)
            print(test_acc)

        # accuracy after training must have increased
        test_acc = policy.accuracy(test_data)
        self.assertGreater(test_acc, 0.85)
