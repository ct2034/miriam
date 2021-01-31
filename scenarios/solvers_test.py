import unittest

import numpy as np
from definitions import INVALID

from scenarios import test_data
from scenarios.solvers import *


class TestSolvers(unittest.TestCase):
    def test_ecbs_success(self):
        # agents that collide in the middle
        res = ecbs(
            test_data.env, test_data.starts_collision,
            test_data.goals_collision)
        self.assertTrue(len(res.keys()) != 0)

    def test_ecbs_paths_no_collision(self):
        # agents that collide in the middle
        paths = ecbs(
            test_data.env, test_data.starts_no_collision,
            test_data.goals_no_collision, return_paths=True)
        test_data.assert_path_equality(
            self, test_data.paths_no_collision, paths)

    def test_ecbs_invalid(self):
        # one agents path is not possible
        starts_invalid = np.array([
            [0, 0],
            [1, 0]  # obstacle
        ])
        goals_invalid = np.array([
            [0, 2],
            [1, 2]  # obstacle
        ])
        self.assertEqual(
            INVALID,
            ecbs(
                test_data.env, starts_invalid, goals_invalid)
        )
        # returns invalid when one agents path is not possible
        starts_invalid = np.array([
            [0, 0],
            [1, 0]  # obstacle
        ])
        goals_invalid = np.array([
            [0, 2],
            [1, 2]  # obstacle
        ])
        self.assertEqual(
            INVALID,
            ecbs(
                test_data.env, starts_invalid, goals_invalid)
        )

    def test_ecbs_deadlocks(self):
        # trying to make deadlocks ...
        starts_deadlocks = np.array([
            [0, 0],
            [0, 1],
            [0, 2],
            [1, 1],
            [2, 2]
        ])
        goals_deadlocks = np.array([
            [2, 0],
            [2, 1],
            [2, 2],
            [0, 0],
            [1, 1]
        ])
        self.assertEqual(
            INVALID,
            ecbs(
                test_data.env, starts_deadlocks, goals_deadlocks)
        )

    def test_icts_paths_no_collision(self):
        # agents that collide in the middle
        paths = icts(
            test_data.env, test_data.starts_no_collision,
            test_data.goals_no_collision, return_paths=True)
        test_data.assert_path_equality(
            self, test_data.paths_no_collision, paths)

    def test_indep_collision(self):
        paths = indep(
            test_data.env, test_data.starts_collision,
            test_data.goals_collision)
        test_data.assert_path_equality(
            self, test_data.paths_collision_indep, paths)

    def test_indep_no_collision(self):
        paths = indep(
            test_data.env, test_data.starts_no_collision,
            test_data.goals_no_collision)
        test_data.assert_path_equality(
            self, test_data.paths_no_collision, paths)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
