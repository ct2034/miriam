#!/usr/bin/env python3

import unittest

import numpy as np

import sim
from agent import Policy, Agent


class TestDecentralizedSim(unittest.TestCase):
    def test_initialize_environment(self):
        env = sim.initialize_environment(10, .5)
        self.assertEqual(env.shape, (10, 10))
        self.assertEqual(np.count_nonzero(env), 50)

    def test_gridmap_to_nx(self):
        env = np.array([[0, 1], [1, 1]])
        g = sim.gridmap_to_nx(env)
        self.assertEqual(len(g), 1)
        self.assertTrue((0, 0) in g)

    def test_initialize_new_agent(self):
        env_zz = np.array([[0, 1], [1, 1]])
        g = sim.gridmap_to_nx(env_zz)
        zero_zero = sim.initialize_new_agent(env_zz, g, [], Policy.RANDOM)
        self.assertTrue((zero_zero.pos == [0, 0]).all())
        self.assertTrue((zero_zero.goal == [0, 0]).all())
        self.assertTrue(zero_zero.policy == Policy.RANDOM)

        env_zo = np.array([[0, 0], [1, 1]])
        g = sim.gridmap_to_nx(env_zo)
        zero_one = sim.initialize_new_agent(env_zo, g, [zero_zero],
                                            Policy.RANDOM)
        self.assertTrue((zero_one.pos == [0, 1]).all())
        self.assertTrue((zero_one.goal == [0, 1]).all())
        self.assertTrue(zero_one.policy == Policy.RANDOM)

    def test_initialize_agents(self):
        env = np.array([[0, 0], [0, 1]])
        g = sim.gridmap_to_nx(env)
        agents = sim.initialize_agents(env, g, 3, Policy.RANDOM)
        self.assertEqual(len(agents), 3)
        self.assertIn((0, 0), map(lambda a: tuple(a.pos), agents))
        self.assertIn((0, 1), map(lambda a: tuple(a.pos), agents))
        self.assertIn((1, 0), map(lambda a: tuple(a.pos), agents))
        self.assertIn((0, 0), map(lambda a: tuple(a.goal), agents))
        self.assertIn((0, 1), map(lambda a: tuple(a.goal), agents))
        self.assertIn((1, 0), map(lambda a: tuple(a.goal), agents))

    def test_plan_path(self):
        env = np.array([[0, 0, 0], [0, 1, 1], [0, 0, 0]])
        g = sim.gridmap_to_nx(env)
        a = Agent(env, g, np.array([0, 2]), Policy.RANDOM)
        a.give_a_goal(np.array([2, 2]))
        p = a.path
        self.assertEqual(len(p), 7)
        self.assertTrue((p[1] == [0, 1]).all())
        self.assertTrue((p[3] == [1, 0]).all())
        self.assertTrue((p[5] == [2, 1]).all())

    @unittest.skip("to be implemented")
    def test_check_for_colissions(self):
        poses = np.array([[0, 0], [1, 0], [1, 1], [1, 2]])
        next_poses = np.array([[0, 0], [0, 0], [1, 2], [1, 1]])
        node_col, edge_col = sim.check_for_colissions(poses, next_poses)
        expected_node_index = (0, 0)
        self.assertIn(expected_node_index, node_col.keys())
        self.assertIn(0, node_col[expected_node_index])
        self.assertIn(1, node_col[expected_node_index])
        expected_edge_index = ((1, 1), (1, 2))
        self.assertIn(expected_edge_index, edge_col.keys())
        self.assertIn(2, edge_col[expected_edge_index])
        self.assertIn(3, edge_col[expected_edge_index])


if __name__ == "__main__":
    unittest.main()
