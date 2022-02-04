import unittest
from itertools import product

import networkx as nx
from planner.policylearn.edge_policy import EdgePolicyModel
from sim.decentralized.agent import env_to_nx

from multi_optim.dagger import *


class ScenarioStateTest(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.graph = nx.Graph()
        for x, y in product(range(5), range(5)):
            self.graph.add_node(x + 5*y, pos=(float(x), float(y)))
            if x > 0:
                self.graph.add_edge(x + 5*y, x - 1 + 5*y)
            if y > 0:
                self.graph.add_edge(x + 5*y, x + 5*(y - 1))
        self.env_nx = env_to_nx(self.graph)
        self.model = EdgePolicyModel(gpu="cpu")

    def test_run_parallel(self):
        starts = [0, 1, 2, 3, 4]
        goals = [20, 21, 22, 23, 24]
        state = ScenarioState(self.graph, starts, goals,
                              self.env_nx, self.model)
        # not finished before running
        self.assertFalse(state.finished)

        # can not observe on never run state
        self.assertRaises(AssertionError, lambda: state.observe())

        state.run()
        # finished after running
        self.assertTrue(state.finished)

        # observation should now be None, because the scenario has no collision
        self.assertIsNone(state.observe())
