#!/usr/bin/env python3

import unittest

import numpy as np

import sim


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
        zero_zero = sim.initialize_new_agent(env_zz, [])
        self.assertTrue((zero_zero == [0, 0]).all())

        env_zo = np.array([[0, 0], [1, 1]])
        zero_one = sim.initialize_new_agent(env_zo, np.array([zero_zero]))
        self.assertTrue((zero_one == [0, 1]).all())

    def test_initialize_agents(self):
        env = np.array([[0, 0], [0, 1]])
        agents = sim.initialize_agents(env, 3)
        self.assertEqual(len(agents), 3)
        self.assertIn([0, 0], agents)
        self.assertIn([0, 1], agents)
        self.assertIn([1, 0], agents)

    def test_plan_path(self):
        env = np.array([[0, 0, 0], [0, 1, 1], [0, 0, 0]])
        g = sim.gridmap_to_nx(env)
        p = sim.plan_path(g, [0, 2], [2, 2])
        self.assertEqual(len(p), 7)
        self.assertTrue((p[1] == [0, 1]).all())
        self.assertTrue((p[3] == [1, 0]).all())
        self.assertTrue((p[5] == [2, 1]).all())

    def test_plan_paths(self):
        env = np.array([[0, 0, 0], [0, 1, 1], [0, 0, 0]])
        g = sim.gridmap_to_nx(env)
        p = sim.plan_paths(g, [[0, 2], [0, 0]], [[2, 2], [0, 2]])
        self.assertEqual(len(p[0]), 7)
        self.assertTrue((p[0][1] == [0, 1]).all())
        self.assertTrue((p[0][3] == [1, 0]).all())
        self.assertTrue((p[0][5] == [2, 1]).all())
        self.assertEqual(len(p[1]), 3)
        self.assertTrue((p[1][0] == [0, 0]).all())
        self.assertTrue((p[1][1] == [0, 1]).all())
        self.assertTrue((p[1][2] == [0, 2]).all())


if __name__ == "__main__":
    unittest.main()
