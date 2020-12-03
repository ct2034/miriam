import unittest

import numpy as np

import scenarios.generators


class TestGenerators(unittest.TestCase):
    def test_generate_like_sim_decentralized_determinism(self):
        (base_env, base_starts, base_goals
         ) = scenarios.generators.like_sim_decentralized(
            10, .5, 10, 0, ignore_cache=True
        )
        # --------

        # everything the same
        (same_env, same_starts, same_goals
         ) = scenarios.generators.like_sim_decentralized(
            10, .5, 10, 0, ignore_cache=True
        )
        self.assertTrue(np.all(base_env == same_env))
        self.assertTrue(np.all(base_starts == same_starts))
        self.assertTrue(np.all(base_goals == same_goals))
        # --------

        # everything different
        (other_env, other_starts, other_goals
         ) = scenarios.generators.like_sim_decentralized(
            10, .5, 10, 1, ignore_cache=True
        )
        self.assertFalse(np.all(base_env == other_env))
        self.assertFalse(np.all(base_starts == other_starts))
        self.assertFalse(np.all(base_goals == other_goals))
        # --------

        # only env different -> all different
        (other_env, other_starts, other_goals
         ) = scenarios.generators.like_sim_decentralized(
            10, .4, 10, 0, ignore_cache=True
        )
        self.assertFalse(np.all(base_env == other_env))
        self.assertFalse(np.all(base_starts == other_starts))
        self.assertFalse(np.all(base_goals == other_goals))

    def test_generate_like_policylearn_gen_determinism(self):
        (base_env, base_starts, base_goals
         ) = scenarios.generators.like_policylearn_gen(
            10, .5, 10, 0, ignore_cache=True
        )
        # --------

        # everything the same
        (same_env, same_starts, same_goals
         ) = scenarios.generators.like_policylearn_gen(
            10, .5, 10, 0, ignore_cache=True
        )
        self.assertTrue(np.all(base_env == same_env))
        self.assertTrue(np.all(base_starts == same_starts))
        self.assertTrue(np.all(base_goals == same_goals))
        # --------

        # everything different
        (other_env, other_starts, other_goals
         ) = scenarios.generators.like_policylearn_gen(
            10, .5, 10, 1, ignore_cache=True
        )
        self.assertFalse(np.all(base_env == other_env))
        self.assertFalse(np.all(base_starts == other_starts))
        self.assertFalse(np.all(base_goals == other_goals))
        # --------

        # only env different -> all different
        (other_env, other_starts, other_goals
         ) = scenarios.generators.like_policylearn_gen(
            10, .4, 10, 0, ignore_cache=True
        )
        self.assertFalse(np.all(base_env == other_env))
        self.assertFalse(np.all(base_starts == other_starts))
        self.assertFalse(np.all(base_goals == other_goals))

    def test_generate_like_policylearn_gen_low_fills(self):
        """tests if generator handles low fill numbers correctly"""
        (env, starts, goals
         ) = scenarios.generators.like_policylearn_gen(
            10, .1, 10, 0, ignore_cache=True
        )
        self.assertEqual(np.count_nonzero(env), 10)  # 10% of 10*10

        (env, starts, goals
         ) = scenarios.generators.like_policylearn_gen(
            10, 0, 10, 0, ignore_cache=True
        )
        self.assertEqual(np.count_nonzero(env), 0)  # 0% of 10*10

    def test_tracing_pathes_in_the_dark_determinism(self):
        (base_env, base_starts, base_goals
         ) = scenarios.generators.tracing_pathes_in_the_dark(
            10, .5, 10, 0
        )
        # --------

        # everything the same
        (same_env, same_starts, same_goals
         ) = scenarios.generators.tracing_pathes_in_the_dark(
            10, .5, 10, 0
        )
        self.assertTrue(np.all(base_env == same_env))
        self.assertTrue(np.all(base_starts == same_starts))
        self.assertTrue(np.all(base_goals == same_goals))

        # everything different
        (other_env, other_starts, other_goals
         ) = scenarios.generators.tracing_pathes_in_the_dark(
            10, .5, 10, 1
        )
        self.assertFalse(np.all(base_env == other_env))
        self.assertFalse(np.all(base_starts == other_starts))
        self.assertFalse(np.all(base_goals == other_goals))
        # --------

        # only env different -> all different
        (other_env, other_starts, other_goals
         ) = scenarios.generators.tracing_pathes_in_the_dark(
            10, .4, 10, 0
        )
        self.assertFalse(np.all(base_env == other_env))
        self.assertFalse(np.all(base_starts == other_starts))
        self.assertFalse(np.all(base_goals == other_goals))

    def test_tracing_pathes_in_the_dark_gen_low_fills(self):
        """tests if generator handles low fill numbers correctly"""
        (env, starts, goals
         ) = scenarios.generators.tracing_pathes_in_the_dark(
            10, .1, 10, 0
        )
        self.assertEqual(np.count_nonzero(env), 10)  # 10% of 10*10

        (env, starts, goals
         ) = scenarios.generators.tracing_pathes_in_the_dark(
            10, 0, 10, 0
        )
        self.assertEqual(np.count_nonzero(env), 0)  # 0% of 10*10

    def test_movingai_loading_maps(self):
        for mapfile in ["Berlin_1_256", "Paris_1_256", "warehouse-10-20-10-2-1", "brc202d"]:
            (env, starts, goals
                ) = scenarios.generators.movingai(mapfile, "even", 0, 100)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
