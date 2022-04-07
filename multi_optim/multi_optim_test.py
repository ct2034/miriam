import unittest
from random import Random
from unittest.mock import patch

import networkx as nx
import torch
from definitions import POS
from planner.policylearn.edge_policy import EdgePolicyModel
from planner.policylearn.edge_policy_test import make_data
from scenarios.test_helper import make_cache_folder_and_set_envvar
from sim.decentralized.agent import Agent
from sim.decentralized.policy import RaisingPolicy, RandomPolicy
from sim.decentralized.runner import to_agent_objects

import multi_optim.state
from multi_optim.multi_optim_run import optimize_policy, sample_trajectory


class MultiOptimTest(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:  # type: ignore
        super().__init__(methodName)
        make_cache_folder_and_set_envvar()
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

    def test_sample_trajectory(self):
        model = EdgePolicyModel(4, 3)
        starts = [0, 4]
        goals = [3, 2]
        rng = Random(1)

        def patch_init(self, graph, _starts, _goals, model, radius):
            print("patch_init")
            self.graph = graph
            self.starts = starts
            self.goals = goals
            self.is_agents_to_consider = None
            self.finished = False
            self.agents = to_agent_objects(
                graph, self.starts, self.goals, radius=radius, rng=rng)
            self.model = model
            if self.agents is None:
                raise RuntimeError("Error in agent generation")
            for a in self.agents:
                a.policy = RaisingPolicy(a)
            self.paths_out = []

        def patch_step(self, _actions):
            """Perform the given actions and return the new state"""
            print(f"{_actions=}")
            print(f"{self.agents[0].pos=}")
            print(f"{self.agents[1].pos=}")
            assert self.agents is not None
            for i_a, a in enumerate(self.agents):
                a.policy = RandomPolicy(a, 1)
                a.start = a.pos
                a.back_to_the_start()
            self.run()

        with patch.object(multi_optim.state.ScenarioState, '__init__',
                          new=patch_init):
            with patch.object(multi_optim.state.ScenarioState, 'step',
                              new=patch_step):
                ds, paths, t = sample_trajectory(
                    seed=0,
                    graph=self.g,
                    n_agents=len(starts),
                    model=model,
                    map_img=((255,),),
                    max_steps=10)
                print(f"{ds=}")
                print(f"{paths=}")
                print(f"{t=}")

        # we have some data from collisions
        self.assertTrue(len(ds) > 0)

        # we have some paths
        assert paths is not None
        self.assertEqual(len(paths), len(starts))
        for i_a, coord_path in enumerate(paths):
            self.assertEqual(len(coord_path), 3)
            _, _, path = coord_path
            self.assertEqual(len(path), 4)
            self.assertEqual(path[0], starts[i_a])
            self.assertEqual(path[3], goals[i_a])
