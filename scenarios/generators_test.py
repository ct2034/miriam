import unittest

import numpy as np

from scenarios.generators import *


class TestGenerators(unittest.TestCase):
    # movingai
    MAPS_WE_CARE_ABOUT = {"Berlin_1_256": [256, 256],
                          "Paris_1_256": [256, 256],
                          "brc202d": [530, 481],
                          "warehouse-10-20-10-2-1": [161, 63]}

    def test_generate_random_gridmap(self):
        w = 999
        h = 100
        gridmap_empty = generate_random_gridmap(w, h, 0)

        assert gridmap_empty.shape[0] == w
        assert gridmap_empty.shape[1] == h
        assert np.max(gridmap_empty) == 0
        assert np.min(gridmap_empty) == 0

        gridmap_half = generate_random_gridmap(w, h, 0.5)

        assert gridmap_half.shape[0] == w
        assert gridmap_half.shape[1] == h
        assert np.max(gridmap_half) == 1
        assert np.min(gridmap_half) == 0

    def test_generate_like_sim_decentralized_determinism(self):
        (base_env, base_starts, base_goals
         ) = like_sim_decentralized(
            10, .5, 10, 0)
        # --------

        # everything the same
        (same_env, same_starts, same_goals
         ) = like_sim_decentralized(
            10, .5, 10, 0)
        self.assertTrue(np.all(base_env == same_env))
        self.assertTrue(np.all(base_starts == same_starts))
        self.assertTrue(np.all(base_goals == same_goals))
        # --------

        # everything different
        (other_env, other_starts, other_goals
         ) = like_sim_decentralized(
            10, .5, 10, 1)
        self.assertFalse(np.all(base_env == other_env))
        self.assertFalse(np.all(base_starts == other_starts))
        self.assertFalse(np.all(base_goals == other_goals))
        # --------

        # only env different -> all different
        (other_env, other_starts, other_goals
         ) = like_sim_decentralized(
            10, .4, 10, 0)
        self.assertFalse(np.all(base_env == other_env))
        self.assertFalse(np.all(base_starts == other_starts))
        self.assertFalse(np.all(base_goals == other_goals))

    def test_generate_like_policylearn_gen_determinism(self):
        (base_env, base_starts, base_goals
         ) = like_policylearn_gen(
            10, .5, 10, 0)
        # --------

        # everything the same
        (same_env, same_starts, same_goals
         ) = like_policylearn_gen(
            10, .5, 10, 0)
        self.assertTrue(np.all(base_env == same_env))
        self.assertTrue(np.all(base_starts == same_starts))
        self.assertTrue(np.all(base_goals == same_goals))
        # --------

        # everything different
        (other_env, other_starts, other_goals
         ) = like_policylearn_gen(
            10, .5, 10, 1)
        self.assertFalse(np.all(base_env == other_env))
        self.assertFalse(np.all(base_starts == other_starts))
        self.assertFalse(np.all(base_goals == other_goals))
        # --------

        # only env different -> all different
        (other_env, other_starts, other_goals
         ) = like_policylearn_gen(
            10, .4, 10, 0)
        self.assertFalse(np.all(base_env == other_env))
        self.assertFalse(np.all(base_starts == other_starts))
        self.assertFalse(np.all(base_goals == other_goals))

    def test_generate_like_policylearn_gen_low_fills(self):
        """tests if generator handles low fill numbers correctly"""
        (env, starts, goals
         ) = like_policylearn_gen(
            10, .1, 10, 0)
        self.assertEqual(np.count_nonzero(env), 10)  # 10% of 10*10

        (env, starts, goals
         ) = like_policylearn_gen(
            10, 0, 10, 0)
        self.assertEqual(np.count_nonzero(env), 0)  # 0% of 10*10

    def test_get_random_next_to_free_pose_or_any_if_full_empty(self):
        # empty map should give any cell
        env_empty = generate_random_gridmap(2, 2, 0)
        pos_empty = get_random_next_to_free_pose_or_any_if_full(
            env_empty)
        self.assertIn(pos_empty[0], [0, 1])
        self.assertIn(pos_empty[1], [0, 1])

    def test_get_random_next_to_free_pose_or_any_if_full_full(self):
        # full map should also give any cell
        env_full = np.full((2, 2), OBSTACLE)
        pos_full = get_random_next_to_free_pose_or_any_if_full(
            env_full)
        self.assertIn(pos_full[0], [0, 1])
        self.assertIn(pos_full[1], [0, 1])

    def test_get_random_next_to_free_pose_or_any_if_full_next(self):
        # checking for the 'next_to' options around center piece
        env_next = np.full((3, 3), OBSTACLE)
        env_next[1, 1] = 0
        pos_next = get_random_next_to_free_pose_or_any_if_full(env_next)
        self.assertIn(tuple(pos_next), [
            (1, 0),
            (2, 1),
            (1, 2),
            (0, 1)
        ])

    def test_tracing_pathes_in_the_dark_determinism(self):
        (base_env, base_starts, base_goals
         ) = tracing_pathes_in_the_dark(
            10, .5, 10, 0)
        # --------

        # everything the same
        (same_env, same_starts, same_goals
         ) = tracing_pathes_in_the_dark(
            10, .5, 10, 0)
        self.assertTrue(np.all(base_env == same_env))
        self.assertTrue(np.all(base_starts == same_starts))
        self.assertTrue(np.all(base_goals == same_goals))

        # everything different
        (other_env, other_starts, other_goals
         ) = tracing_pathes_in_the_dark(
            10, .5, 10, 1)
        self.assertFalse(np.all(base_env == other_env))
        self.assertFalse(np.all(base_starts == other_starts))
        self.assertFalse(np.all(base_goals == other_goals))
        # --------

        # only env different -> all different
        (other_env, other_starts, other_goals
         ) = tracing_pathes_in_the_dark(
            10, .4, 10, 0)
        self.assertFalse(np.all(base_env == other_env))
        self.assertFalse(np.all(base_starts == other_starts))
        self.assertFalse(np.all(base_goals == other_goals))

    def test_tracing_pathes_in_the_dark_gen_low_fills(self):
        """tests if generator handles low fill numbers correctly"""
        (env, starts, goals
         ) = tracing_pathes_in_the_dark(
            10, .1, 10, 0)
        self.assertEqual(np.count_nonzero(env), 10)  # 10% of 10*10

        (env, starts, goals
         ) = tracing_pathes_in_the_dark(
            10, 0, 10, 0)
        self.assertEqual(np.count_nonzero(env), 0)  # 0% of 10*10

    # movingai ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def test_movingai_read_mapfile(self):
        for mapname in self.MAPS_WE_CARE_ABOUT.keys():
            mapfile = 'scenarios/movingai/mapf-map/' + mapname + '.map'
            env = movingai_read_mapfile(mapfile)
            assert len(env.shape) == 2
            assert env.shape[0] == self.MAPS_WE_CARE_ABOUT[mapname][0]
            assert env.shape[1] == self.MAPS_WE_CARE_ABOUT[mapname][1]

    def test_movingai_loading_maps(self):
        for mapfile in self.MAPS_WE_CARE_ABOUT:
            (env, starts, goals
             ) = movingai(mapfile, "even", 0, 100)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
