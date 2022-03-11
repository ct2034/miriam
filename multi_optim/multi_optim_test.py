import unittest
from random import Random

import networkx as nx
import torch
from definitions import POS
from planner.policylearn.edge_policy import EdgePolicyModel
from planner.policylearn.edge_policy_test import make_data
from sim.decentralized.agent import Agent

from multi_optim.multi_optim_run import find_collisions, optimize_policy


class MultiOptimTest(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:  # type: ignore
        super().__init__(methodName)
        self.g = nx.Graph()
        self.g.add_edges_from([
            (0, 1),
            (1, 2),
            (1, 3),
            (1, 4),
            (3, 4)])
        nx.set_node_attributes(self.g, {
            0: (0., 0.),
            1: (1., 1.),
            2: (0., 2.),
            3: (2., 2.),
            4: (2., 0.)}, POS)
        self.radius = .3

        #   2   3
        #    \ /|
        #     1 |
        #    / \|
        #   0   4

    def test_find_collisions(self):
        agents = [
            Agent(self.g, 0, radius=self.radius),
            Agent(self.g, 4, radius=self.radius),
            Agent(self.g, 3, radius=self.radius)]
        agents[0].give_a_goal(3)
        agents[1].give_a_goal(2)
        agents[2].give_a_goal(4)
        collisions = find_collisions(agents)
        self.assertEqual(len(collisions), 1)
        self.assertEqual(list(collisions.keys())[0][0], 1)  # node
        self.assertEqual(list(collisions.keys())[0][1], 1)  # t
        self.assertIn(0, collisions[(1, 1)])  # agent
        self.assertIn(1, collisions[(1, 1)])  # agent

    def test_optimize_policy(self):
        # this is analog to edge_policy_test.py / test_edge_policy_learn
        rng = Random(0)
        torch.manual_seed(0)
        num_node_features = 2
        conv_channels = 3
        n_nodes = 2
        n_epochs = 20
        n_data = 10
        policy = EdgePolicyModel(
            num_node_features, conv_channels)
        optimizer = torch.optim.Adam(policy.parameters(), lr=.01)
        datas = []

        # test data
        test_data = [(
            make_data(rng, num_node_features, n_nodes),
            {x: x for x in range(n_nodes)}) for _ in range(10)]
        test_acc = policy.accuracy(test_data)
        self.assertLess(test_acc, 0.6)

        # train
        for _ in range(n_epochs):
            for _ in range(n_data):
                datas.append(make_data(rng, num_node_features, n_nodes))
            policy, loss = optimize_policy(
                policy, batch_size=5, optimizer=optimizer, epds=datas)
            test_acc = policy.accuracy(test_data)
            print(test_acc)

        # accuracy after training must have increased
        test_acc = policy.accuracy(test_data)
        self.assertGreater(test_acc, 0.85)
