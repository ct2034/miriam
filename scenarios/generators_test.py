import unittest

import numpy as np
from numpy.core.numeric import count_nonzero

from scenarios.generators import *
from scenarios.test_helper import is_connected


class TestGenerators(unittest.TestCase):
    # movingai
    MAPS_WE_CARE_ABOUT = {"Berlin_1_256": [256, 256],
                          "Paris_1_256": [256, 256],
                          "brc202d": [530, 481],
                          "warehouse-10-20-10-2-1": [161, 63]}

    def test_generate_random_gridmap(self):
        w = 999
        h = 100
        rng = random.Random(0)
        gridmap_empty = generate_walls_gridmap(w, h, 0, rng)

        assert gridmap_empty.shape[0] == w
        assert gridmap_empty.shape[1] == h
        assert np.max(gridmap_empty) == 0
        assert np.min(gridmap_empty) == 0

        gridmap_half = generate_walls_gridmap(w, h, 0.5, rng)

        assert gridmap_half.shape[0] == w
        assert gridmap_half.shape[1] == h
        assert np.max(gridmap_half) == 1
        assert np.min(gridmap_half) == 0

    def test_generate_like_sim_decentralized_determinism(self):
        (base_env, base_starts, base_goals
         ) = random_fill(
            10, .5, 10, random.Random(0))
        # --------

        # everything the same
        (same_env, same_starts, same_goals
         ) = random_fill(
            10, .5, 10, random.Random(0))
        self.assertTrue(np.all(base_env == same_env))
        self.assertTrue(np.all(base_starts == same_starts))
        self.assertTrue(np.all(base_goals == same_goals))
        # --------

        # everything different
        (other_env, other_starts, other_goals
         ) = random_fill(
            10, .5, 10, random.Random(1))
        self.assertFalse(np.all(base_env == other_env))
        self.assertFalse(np.all(base_starts == other_starts))
        self.assertFalse(np.all(base_goals == other_goals))
        # --------

        # only env different -> all different
        (other_env, other_starts, other_goals
         ) = random_fill(
            10, .4, 10, random.Random(0))
        self.assertFalse(np.all(base_env == other_env))
        self.assertFalse(np.all(base_starts == other_starts))
        self.assertFalse(np.all(base_goals == other_goals))

    def test_generate_like_policylearn_gen_determinism(self):
        (base_env, base_starts, base_goals
         ) = walls(
            10, .5, 10, random.Random(0))
        # --------

        # everything the same
        (same_env, same_starts, same_goals
         ) = walls(
            10, .5, 10, random.Random(0))
        self.assertTrue(np.all(base_env == same_env))
        self.assertTrue(np.all(base_starts == same_starts))
        self.assertTrue(np.all(base_goals == same_goals))
        # --------

        # everything different
        (other_env, other_starts, other_goals
         ) = walls(
            10, .5, 10, random.Random(1))
        self.assertFalse(np.all(base_env == other_env))
        self.assertFalse(np.all(base_starts == other_starts))
        self.assertFalse(np.all(base_goals == other_goals))
        # --------

        # only env different -> all different
        (other_env, other_starts, other_goals
         ) = walls(
            10, .4, 10, random.Random(0))
        self.assertFalse(np.all(base_env == other_env))
        self.assertFalse(np.all(base_starts == other_starts))
        self.assertFalse(np.all(base_goals == other_goals))

    def test_generate_like_policylearn_gen_low_fills(self):
        """tests if generator handles low fill numbers correctly"""
        (env, starts, goals
         ) = walls(
            10, .1, 10, random.Random(0))
        self.assertEqual(np.count_nonzero(env), 10)  # 10% of 10*10

        (env, starts, goals
         ) = walls(
            10, 0, 10, random.Random(0))
        self.assertEqual(np.count_nonzero(env), 0)  # 0% of 10*10

    def test_get_random_next_to_free_pose_or_any_if_full_empty(self):
        # empty map should give any cell
        rng = random.Random(0)
        env_empty = generate_walls_gridmap(2, 2, 0, rng)
        pos_empty = get_random_next_to_free_pose_or_any_if_full(
            env_empty, rng)
        self.assertIn(pos_empty[0], [0, 1])
        self.assertIn(pos_empty[1], [0, 1])

    def test_get_random_next_to_free_pose_or_any_if_full_full(self):
        # full map should also give any cell
        env_full = np.full((2, 2), OBSTACLE)
        pos_full = get_random_next_to_free_pose_or_any_if_full(
            env_full, random.Random(0))
        self.assertIn(pos_full[0], [0, 1])
        self.assertIn(pos_full[1], [0, 1])

    def test_get_random_next_to_free_pose_or_any_if_full_next(self):
        # checking for the 'next_to' options around center piece
        env_next = np.full((3, 3), OBSTACLE)
        env_next[1, 1] = 0
        pos_next = get_random_next_to_free_pose_or_any_if_full(
            env_next, random.Random(0))
        self.assertIn(tuple(pos_next), [
            (1, 0),
            (2, 1),
            (1, 2),
            (0, 1)
        ])

    def test_tracing_pathes_in_the_dark_determinism(self):
        (base_env, base_starts, base_goals
         ) = tracing_pathes_in_the_dark(
            10, .5, 10, random.Random(0))
        # --------

        # everything the same
        (same_env, same_starts, same_goals
         ) = tracing_pathes_in_the_dark(
            10, .5, 10, random.Random(0))
        self.assertTrue(np.all(base_env == same_env))
        self.assertTrue(np.all(base_starts == same_starts))
        self.assertTrue(np.all(base_goals == same_goals))

        # everything different
        (other_env, other_starts, other_goals
         ) = tracing_pathes_in_the_dark(
            10, .5, 10, random.Random(1))
        self.assertFalse(np.all(base_env == other_env))
        self.assertFalse(np.all(base_starts == other_starts))
        self.assertFalse(np.all(base_goals == other_goals))
        # --------

        # only env different -> all different
        (other_env, other_starts, other_goals
         ) = tracing_pathes_in_the_dark(
            10, .4, 10, random.Random(0))
        self.assertFalse(np.all(base_env == other_env))
        self.assertFalse(np.all(base_starts == other_starts))
        self.assertFalse(np.all(base_goals == other_goals))

    def test_tracing_pathes_in_the_dark_gen_low_fills(self):
        """tests if generator handles low fill numbers correctly"""
        (env, starts, goals
         ) = tracing_pathes_in_the_dark(
            10, .1, 10, random.Random(0))
        self.assertEqual(np.count_nonzero(env), 10)  # 10% of 10*10

        (env, starts, goals
         ) = tracing_pathes_in_the_dark(
            10, 0, 10, random.Random(0))
        self.assertEqual(np.count_nonzero(env), 0)  # 0% of 10*10

    def test_tracing_pathes_in_the_dark_radomism(self):
        """tests if generator makes actually random changes between agents"""
        (env, starts, goals
         ) = tracing_pathes_in_the_dark(
            10, 0, 10, random.Random(0))
        problematic = np.array(starts)[1:, :]
        self.assertFalse(all(problematic[:, 1] == np.arange(9)))
        self.assertFalse(all(problematic[:, 0] == 5))

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

    def test_can_area_be_set_true(self):
        # completely free area can be set
        free = np.array([
            [1, 1, 1],
            [0, 0, 0],
            [0, 0, 0]
        ])
        self.assertTrue(can_area_be_set(free))

        # half free area can be set
        half_free = np.array([
            [1, 1, 1],
            [0, 0, 1],
            [0, 0, 0]
        ])
        self.assertTrue(can_area_be_set(half_free))

    def test_can_area_be_set_false(self):
        # some two changes can not be set
        some = np.array([
            [1, 0, 1],
            [1, 0, 0],
            [1, 1, 1]
        ])
        self.assertFalse(can_area_be_set(some))

        # connecting two walls can not be set
        two_walls = np.array([
            [0, 1, 0],
            [0, 0, 0],
            [0, 1, 0]
        ])
        self.assertFalse(can_area_be_set(two_walls))

    # building walls ----------------------------------------------------------
    def test_can_area_be_set_raises(self):
        # full area raises assertion
        full = np.full((3, 3), OBSTACLE)
        self.assertRaises(AssertionError, lambda: can_area_be_set(full))
        # wrong width raises assertion
        wrong_width = np.full((2, 3), FREE)
        self.assertRaises(AssertionError, lambda: can_area_be_set(wrong_width))
        # wrong height raises assertion
        wrong_height = np.full((3, 2), FREE)
        self.assertRaises(
            AssertionError, lambda: can_area_be_set(wrong_height))

    def test_can_be_set_true(self):
        # completely free area can be set
        free = np.array([
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0]
        ])
        self.assertTrue(can_be_set(free, [0, 1]))
        self.assertTrue(can_be_set(free, [1, 1]))
        self.assertTrue(can_be_set(free, [2, 1]))

        # half free area can be set
        half_free = np.array([
            [1, 1, 1],
            [0, 0, 1],
            [0, 0, 0]
        ])
        self.assertTrue(can_be_set(half_free, [1, 1]))

    def test_can_be_set_false(self):
        # some two changes can not be set
        some = np.array([
            [1, 0, 1, 0],
            [1, 0, 0, 0],
            [1, 1, 1, 1]
        ])
        self.assertFalse(can_be_set(some, [1, 1]))
        self.assertFalse(can_be_set(some, [1, 2]))
        self.assertFalse(can_be_set(some, [1, 3]))

        # connecting two walls can not be set
        two_walls = np.array([
            [0, 1, 0],
            [0, 0, 0],
            [0, 1, 0]
        ])
        self.assertFalse(can_be_set(two_walls, [1, 1]))

    def test_can_be_set_raises(self):
        # wrong pos dims
        full = np.full((3, 3), OBSTACLE)
        self.assertRaises(AssertionError, lambda: can_be_set(full, [1, 1, 1]))
        # pos out ot right
        self.assertRaises(AssertionError, lambda: can_be_set(full, [1, 3]))
        # pos out to bottom
        self.assertRaises(AssertionError, lambda: can_be_set(full, [3, 1]))

    def test_corridor_with_passing(self):
        size = 10
        # test if correct dimensions are returned
        (env, starts, goals) = corridor_with_passing(
            size, 0, 2, random.Random(0))
        self.assertEqual(env.shape, (10, 10))
        self.assertEqual(starts.shape, (2, 2))
        self.assertEqual(goals.shape, (2, 2))
        # at least some cells need to be 0
        self.assertGreater(np.count_nonzero(env == 0), 0)
        # at least some cells need to be 1
        self.assertGreater(np.count_nonzero(env == 1), 0)

        # test if there is exactly one passing point
        path_len = abs(starts[0, 0] - goals[0, 0]) + \
            abs(starts[0, 1] - goals[0, 1]) + 1
        self.assertEqual(np.count_nonzero(env == FREE),
                         path_len+1)  # +1 for passing point
        self.assertEqual(np.count_nonzero(env == OBSTACLE),
                         size**2 - path_len - 1)

        # make sure all free cells are connected
        self.assertTrue(is_connected(env))

    # arena with crossing -----------------------------------------------------
    def test_arena_with_crossing_no_duplicate_starts_or_goals(self):
        rng = random.Random(0)
        for _ in range(10):
            # test if correct dimensions are returned
            (env, starts, goals
             ) = arena_with_crossing(10, 0, 8, rng)
            self.assertEqual(env.shape, (10, 10))
            self.assertEqual(starts.shape, (8, 2))
            self.assertEqual(goals.shape, (8, 2))
            # all cells are zero here
            self.assertEqual(np.count_nonzero(env), 0)

            # make sure all free cells are connected
            self.assertTrue(is_connected(env))

            # make sure there are no duplicates
            self.assertTrue(starts.shape == np.unique(starts, axis=0).shape)
            self.assertTrue(goals.shape == np.unique(goals, axis=0).shape)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
