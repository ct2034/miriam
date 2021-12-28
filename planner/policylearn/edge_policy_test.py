import unittest

import torch
from planner.policylearn.edge_policy import EdgePolicyModel


class EdgePolicyTest(unittest.TestCase):
    def test_edge_policy(self):
        n_nodes = 5
        num_node_features = 2
        conv_channels = 2
        policy = EdgePolicyModel(num_node_features, conv_channels)
        x = torch.randn(n_nodes, num_node_features)
        edge_index = torch.tensor([
            [0, 0, 0, 0],
            [1, 2, 3, 4]
        ])
        pos = torch.tensor([
            [0, 0],
            [1, 0],
            [0, 1],
            [-1, 0],
            [0, -1]
        ])
        node = 0
        score, targets = policy(x, edge_index, pos, node)
        self.assertEqual(score.shape, (4,))
        self.assertEqual(targets.shape, (4,))
