import unittest
from cgi import test
from itertools import product
from random import Random
from timeit import repeat

import torch
from planner.policylearn.edge_policy import EdgePolicyModel
from torch_geometric.data import Data


class EdgePolicyTest(unittest.TestCase):
    def test_edge_policy(self):
        n_nodes = 5
        num_node_features = 2
        conv_channels = 2
        policy = EdgePolicyModel(num_node_features, conv_channels)
        x = torch.randn(n_nodes, num_node_features)
        x[0, 0] = 1  # node 1 is our node
        edge_index = torch.tensor([
            [0, 0, 0, 0],
            [1, 2, 3, 4]
        ])
        score, targets = policy(x, edge_index)
        self.assertEqual(score.shape, (4,))
        self.assertEqual(targets.shape, (4,))

    @unittest.skip(reason="lets make this work properly, first")
    def test_edge_policy_learn(self):
        rng = Random(0)
        num_node_features = 1
        conv_channels = 1
        n_nodes = 2
        n_epochs = 10
        n_data = 10
        policy = EdgePolicyModel(num_node_features, conv_channels)
        optimizer = torch.optim.SGD(policy.parameters(), lr=.1)
        datas = []

        # test data
        big_from_small = {n: n for n in range(n_nodes)}
        test_data = self.make_data(rng, num_node_features, n_nodes)
        pred = policy.predict(
            test_data.x, test_data.edge_index, big_from_small)
        self.assertNotEqual(pred, test_data.y)

        # train
        for _ in range(n_epochs):
            for _ in range(n_data):
                datas.append(self.make_data(rng, num_node_features, n_nodes))
            policy.learn(datas, optimizer)
            print(policy.accuracy(datas, [big_from_small] * len(datas)))
        self.assertEqual(pred, test_data.y)

    def make_data(self, rng, num_node_features, n_nodes):
        node = rng.randint(0, n_nodes-1)
        data_x = torch.tensor(
            [[rng.random() for _ in range(num_node_features)]
             for _ in range(n_nodes)]
        )
        data_x[node, 0] = 1.0  # learn from this
        data_edge_index = torch.tensor(
            [[a, b] for a, b in product(range(n_nodes), repeat=2)]
        ).T
        data_y = node
        data = Data(
            x=data_x,
            edge_index=data_edge_index,
            y=data_y)

        return data
