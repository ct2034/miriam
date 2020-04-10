#!/usr/bin/env python3

import unittest

import numpy as np

import sim


class TestDecentralizedSim(unittest.TestCase):
    def test_initialize_environment(self):
        env = sim.initialize_environment(10, .5)
        self.assertEqual(env.shape, (10, 10))
        self.assertEqual(np.count_nonzero(env), 50)

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

if __name__ == "__main__":
    unittest.main()
