import unittest
from itertools import product
from math import pi

import networkx as nx
from planner.policylearn.edge_policy import EdgePolicyModel
from sim.decentralized.agent import env_to_nx

from multi_optim.dagger import *


class ScenarioStateTest(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:  # type: ignore
        super().__init__(methodName)

        # square graph
        self.sq_graph = nx.Graph()
        for x, y in product(range(5), range(5)):
            self.sq_graph.add_node(x + 5*y, pos=(float(x), float(y)))
            if x > 0:
                self.sq_graph.add_edge(x + 5*y, x - 1 + 5*y)
            if y > 0:
                self.sq_graph.add_edge(x + 5*y, x + 5*(y - 1))
        self.sq_env_nx = env_to_nx(self.sq_graph)

        # graph with passing point
        self.pass_graph = nx.Graph()
        self.pass_graph.add_node(0, pos=(0., 0.))
        self.pass_graph.add_node(1, pos=(1., 0.))
        self.pass_graph.add_node(2, pos=(1., 1.))
        self.pass_graph.add_node(3, pos=(2., 0.))
        self.pass_graph.add_node(4, pos=(3., 0.))
        self.pass_graph.add_edge(0, 1)
        self.pass_graph.add_edge(0, 2)
        self.pass_graph.add_edge(1, 3)
        self.pass_graph.add_edge(1, 2)
        self.pass_graph.add_edge(3, 4)
        self.pass_env_nx = env_to_nx(self.pass_graph)

        # any model
        self.model = EdgePolicyModel(gpu="cpu")

    def test_run_parallel(self):
        """Test running the scenario with parallel paths."""
        starts = [0, 1, 2, 3, 4]
        goals = [20, 21, 22, 23, 24]
        state = ScenarioState(self.sq_graph, starts, goals,
                              self.sq_env_nx, self.model)
        # not finished before running
        self.assertFalse(state.finished)

        # can not observe on never run state
        self.assertRaises(AssertionError, lambda: state.observe())

        state.run()
        # finished after running
        self.assertTrue(state.finished)

        # observation should now be None, because the scenario has no collision
        self.assertIsNone(state.observe())

    def test_run_pass_with_collision(self):
        """Test running the scenario with a collision."""
        starts = [0, 3]
        goals = [3, 0]
        state = ScenarioState(self.pass_graph, starts, goals,
                              self.pass_env_nx, self.model)
        # not finished before running
        self.assertFalse(state.finished)

        # can not observe on never run state
        self.assertRaises(AssertionError, lambda: state.observe())

        state.run()
        # not finished after running
        self.assertFalse(state.finished)

        # there should be an observation now
        observation = state.observe()
        self.assertIsNotNone(observation)

    def test_observe_pass_scenario(self):
        """Test observation in the scenario with a collision."""
        # preparation
        starts = [0, 3]
        goals = [3, 0]
        state = ScenarioState(self.pass_graph, starts, goals,
                              self.pass_env_nx, self.model)
        state.run()

        # check state after running
        # the agents should not have moved
        self.assertFalse(state.finished)
        self.assertEqual(state.agents[0].pos, (0))
        self.assertEqual(state.agents[1].pos, (3))
        self.assertIn(0, state.is_agents_to_consider)
        self.assertIn(1, state.is_agents_to_consider)

        # there should be an observation now
        observation = state.observe()
        self.assertIsNotNone(observation)
        self.assertEqual(len(observation), 2)
        data_0, big2sml_0 = observation[0]
        data_1, big2sml_1 = observation[1]

        # check the data
        for d in [data_0, data_1]:
            self.assertEqual(d.num_nodes, self.pass_graph.number_of_nodes())
            # made it undirected
            self.assertEqual(d.num_edges, 2*self.pass_graph.number_of_edges())
            self.assertEqual(d.x.shape[0], self.pass_graph.number_of_nodes())
            self.assertEqual(d.x.shape[1], self.model.num_node_features)
        # 1. path layer ...
        #    current poses for each agent
        self.assertEqual(data_0.x[0, 0], 1.)
        self.assertEqual(data_1.x[3, 0], 1.)
        #    intermediate poses for each agent
        self.assertEqual(data_0.x[1, 0], 1.1)
        self.assertEqual(data_1.x[1, 0], 1.1)
        #    goal poses for each agent
        self.assertEqual(data_0.x[3, 0], 1.2)
        self.assertEqual(data_1.x[0, 0], 1.2)
        # 2. other path layer ...
        #    current poses for each agent
        self.assertEqual(data_0.x[3, 1], 1.0)
        self.assertEqual(data_1.x[0, 1], 1.0)
        #    intermediate poses for each agent
        self.assertEqual(data_0.x[1, 1], 1.1)
        self.assertEqual(data_1.x[1, 1], 1.1)
        #    goal poses for each agent
        self.assertEqual(data_0.x[0, 1], 1.2)
        self.assertEqual(data_1.x[3, 1], 1.2)
        # 3. relative distance layer ...
        #    for own pose
        self.assertEqual(data_0.x[0, 2], 0.)
        self.assertEqual(data_1.x[3, 2], 0.)
        #    for node 4
        self.assertEqual(data_0.x[4, 2], 3.)
        self.assertEqual(data_1.x[4, 2], 1.)
        # 4. relative angle layer ...
        #    for own pose
        self.assertEqual(data_0.x[0, 3], 0.)   # own pose always 0
        self.assertEqual(data_1.x[3, 3], 0.)   # own pose always 0
        #    for node 4
        self.assertEqual(data_0.x[4, 3], 0.)   # in front of me
        self.assertEqual(data_1.x[4, 3], -pi)  # behind me
        #    for node 2
        self.assertAlmostEqual(data_0.x[2, 3].item(), pi / 4, places=3)
        self.assertAlmostEqual(data_1.x[2, 3].item(), -pi / 4, places=3)

        # network is small enough, all nodes should be in the observation
        for b2s in [big2sml_0, big2sml_1]:
            self.assertEqual(len(b2s), self.pass_graph.number_of_nodes())
            for k, v in b2s.items():
                self.assertEqual(k, v)

        # now we make a step
        state.step({0: 2, 1: 1})  # agent 0 moves to node 2, agent 1 to node 1

        # check state after running
        self.assertTrue(state.finished)
        self.assertEqual(state.agents[0].pos, (3))
        self.assertEqual(state.agents[1].pos, (0))
        self.assertIsNone(state.observe())
        self.assertIsNone(state.is_agents_to_consider)

        # check pathses we took to be correct
        self.assertEqual(state.paths_out[0], [0, 2, 1, 3])
