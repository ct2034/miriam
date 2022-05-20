import unittest
from itertools import product
from math import atan2, pi
from random import Random

import networkx as nx
import numpy as np
from learn.delaunay_benchmark.delaunay_implementations import (
    read_map, run_delaunay_libpysal)
from planner.policylearn.edge_policy_graph_utils import *
from scenarios.test_helper import make_cache_folder_and_set_envvar
from sim.decentralized.agent import Agent
from sim.decentralized.runner import to_agent_objects


class TestEdgePolicyGraphUtils(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestEdgePolicyGraphUtils, self).__init__(*args, **kwargs)
        make_cache_folder_and_set_envvar()
        self.env = nx.Graph()
        for x, y in product(range(7), range(7)):
            self.env.add_node(x + y * 7, pos=(float(x), float(y)))
            if x > 0:
                self.env.add_edge(x + y * 7, x - 1 + y * 7)
            if y > 0:
                self.env.add_edge(x + y * 7, x + (y - 1) * 7)
        self.agents = [
            Agent(self.env, pos=24, radius=.3),
            Agent(self.env, pos=25, radius=.3)
        ]
        self.agents[0].give_a_goal(27)
        self.agents[1].give_a_goal(21)

    def test_t_to_data(self):
        self.assertEqual(t_to_data(99, 99), 1)
        self.assertEqual(t_to_data(99, 98), 1.1)
        self.assertEqual(t_to_data(99, 97), 1.2)
        self.assertEqual(t_to_data(99, 100), 0)

    def test_agents_to_data(self):
        pos = nx.get_node_attributes(self.env, "pos")
        data, big_from_small = agents_to_data(self.agents, 0)
        small_from_big = {v: k for k, v in big_from_small.items()}

        # make sure big_from_small is correct
        expected_nodes_big = [
            45,
            37, 38, 39,
            29, 30, 31, 32, 33,
            21, 22, 23, 24, 25, 26, 27,
            15, 16, 17, 18, 19,
            9, 10, 11,
            3]
        self.assertEqual(len(big_from_small), len(expected_nodes_big))
        for n in expected_nodes_big:
            self.assertIn(n, list(big_from_small.values()))

        # check data
        # 1. own path
        self.assertEqual(data.x[small_from_big[24], 0], 1.0)
        self.assertEqual(data.x[small_from_big[25], 0], 1.1)
        self.assertEqual(data.x[small_from_big[26], 0], 1.2)
        self.assertEqual(data.x[small_from_big[27], 0], 1.3)
        for n in expected_nodes_big:
            if n in [24, 25, 26, 27]:
                continue
            self.assertEqual(data.x[small_from_big[n], 0], 0.0)
        # 2. others path
        self.assertEqual(data.x[small_from_big[25], 1], 1.0)
        self.assertEqual(data.x[small_from_big[24], 1], 1.1)
        self.assertEqual(data.x[small_from_big[23], 1], 1.2)
        self.assertEqual(data.x[small_from_big[22], 1], 1.3)
        self.assertEqual(data.x[small_from_big[21], 1], 1.4)
        for n in expected_nodes_big:
            if n in [24, 25, 23, 22, 21]:
                continue
            self.assertEqual(data.x[small_from_big[n], 1], 0.0)
        # 3. relative distance
        for n in expected_nodes_big:
            d = torch.linalg.norm(torch.tensor(pos[n]) - torch.tensor(pos[24]))
            self.assertAlmostEqual(data.x[small_from_big[n], 2], d, places=3)
        # 4. relative angle
        for n in expected_nodes_big:
            if n == 24:
                a = 0
            else:
                a = atan2(pos[n][1] - pos[24][1], pos[n][0] - pos[24][0])
            self.assertAlmostEqual(data.x[small_from_big[n], 3], a, places=3)

    def test_angles_of_random_graphs(self):
        n_nodes = 32
        n_smpls = 100
        n_agents = 2
        rng = Random(0)
        map_fname: str = "roadmaps/odrm/odrm_eval/maps/plain.png"
        map_img = read_map(map_fname)
        for _ in range(n_smpls):
            pos = np.array(
                [(rng.randint(0, map_img.shape[0]-1),
                    rng.randint(0, map_img.shape[1]-1))
                    for _ in range(n_nodes)],
                dtype=np.int32)
            g = run_delaunay_libpysal(pos, map_img)
            starts = rng.sample(range(g.number_of_nodes()), k=n_agents)
            goals = rng.sample(range(g.number_of_nodes()), k=n_agents)
            agents = to_agent_objects(g, starts, goals, radius=.1)
            assert agents is not None
            data, big_from_small = agents_to_data(agents, 0)
            for n in range(n_nodes):
                self.assertGreaterEqual(data.x[n, 3], -pi/2)
                self.assertLessEqual(data.x[n, 3], pi/2)

    def test_get_optimal_edge(self):
        agents = self.agents.copy()
        agents[0].give_a_goal(34)
        agents[1].give_a_goal(14)

        opt_node_0 = get_optimal_edge(agents, 0)
        opt_node_1 = get_optimal_edge(agents, 1)

        self.assertEqual(opt_node_0, 31)
        self.assertEqual(opt_node_1, 24)
