import unittest

import numpy as np

import scenarios.generators


class TestGenerators(unittest.TestCase):
    def test_generate_like_sim_decentralized_determinism(self):
        base_env, base_agents = scenarios.generators.like_sim_decentralized(
            10, .5, 10, 0
        )
        # --------

        same_env, same_agents = scenarios.generators.like_sim_decentralized(
            10, .5, 10, 0
        )
        self.assertTrue(np.all(base_env == same_env))
        self.assertEqual(base_agents, same_agents)
        # --------

        other_env, other_agents = scenarios.generators.like_sim_decentralized(
            10, .5, 10, 1
        )
        self.assertFalse(np.all(base_env == other_env))
        self.assertNotEqual(base_agents, other_agents)
        # --------

        same_env, other_agents = scenarios.generators.like_sim_decentralized(
            10, .5, 9, 0
        )
        self.assertTrue(np.all(base_env == same_env))
        self.assertNotEqual(base_agents, other_agents)
