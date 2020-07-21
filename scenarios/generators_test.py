import unittest

import numpy as np

import scenarios.generators


class TestGenerators(unittest.TestCase):
    def test_generate_like_sim_decentralized_determinism(self):
        (base_env, base_starts, base_goals
         ) = scenarios.generators.like_sim_decentralized(
            10, .5, 10, 0
        )
        # --------

        # everything the same
        (same_env, same_starts, same_goals
         ) = scenarios.generators.like_sim_decentralized(
            10, .5, 10, 0
        )
        self.assertTrue(np.all(base_env == same_env))
        self.assertTrue(np.all(base_starts == same_starts))
        self.assertTrue(np.all(base_goals == same_goals))
        # --------

        # everything different
        (other_env, other_starts, other_goals
         ) = scenarios.generators.like_sim_decentralized(
            10, .5, 10, 1
        )
        self.assertFalse(np.all(base_env == other_env))
        self.assertFalse(np.all(base_starts == other_starts))
        self.assertFalse(np.all(base_goals == other_goals))
        # --------

        # only env different -> all different
        (other_env, other_starts, other_goals
         ) = scenarios.generators.like_sim_decentralized(
            10, .4, 10, 0
        )
        self.assertFalse(np.all(base_env == other_env))
        self.assertFalse(np.all(base_starts == other_starts))
        self.assertFalse(np.all(base_goals == other_goals))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
