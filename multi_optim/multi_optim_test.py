import unittest

import pytest
from definitions import POS
from sim.decentralized.agent import Agent

from multi_optim.multi_optim import *


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

        #   2   3
        #    \ /|
        #     1 |
        #    / \|
        #   0   4

    def test_find_collisions(self):
        agents = [
            Agent(self.g, 0),
            Agent(self.g, 4),
            Agent(self.g, 3)]
        agents[0].give_a_goal(3)
        agents[1].give_a_goal(2)
        agents[2].give_a_goal(4)
        collisions = find_collisions(agents)
        self.assertEqual(len(collisions), 1)
        self.assertEqual(list(collisions.keys())[0][0], 1)  # node
        self.assertEqual(list(collisions.keys())[0][1], 1)  # t
        self.assertIn(0, collisions[(1, 1)])  # agent
        self.assertIn(1, collisions[(1, 1)])  # agent
