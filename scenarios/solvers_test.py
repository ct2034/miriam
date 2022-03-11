import unittest
import pytest

import numpy as np
from definitions import INVALID

from scenarios import test_helper
from scenarios.solvers import *


class TestSolvers(unittest.TestCase):
    def test_ecbs_success(self):
        # agents that collide in the middle
        res = ecbs(
            test_helper.env, test_helper.starts_collision,
            test_helper.goals_collision)
        self.assertNotEqual(res, INVALID)  # successful
        self.assertTrue(len(res.keys()) != 0)

    def test_ecbs_paths_no_collision(self):
        # agents that collide in the middle
        paths = ecbs(
            test_helper.env, test_helper.starts_no_collision,
            test_helper.goals_no_collision, return_paths=True)
        self.assertNotEqual(paths, INVALID)  # successful
        test_helper.assert_path_equality(
            self, test_helper.paths_no_collision, paths)

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
                test_helper.env, starts_invalid, goals_invalid)
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
                test_helper.env, starts_invalid, goals_invalid)
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
                test_helper.env, starts_deadlocks, goals_deadlocks)
        )

    def test_icts_paths_no_collision(self):
        # agents that collide in the middle
        paths = icts(
            test_helper.env, test_helper.starts_no_collision,
            test_helper.goals_no_collision, return_paths=True)
        test_helper.assert_path_equality(
            self, test_helper.paths_no_collision, paths)

    @pytest.mark.skip  # not working on this right now
    def test_indep_collision(self):
        paths = indep(
            test_helper.env, test_helper.starts_collision,
            test_helper.goals_collision)
        test_helper.assert_path_equality(
            self, test_helper.paths_collision_indep, paths)

    @pytest.mark.skip  # not working on this right now
    def test_indep_no_collision(self):
        paths = indep(
            test_helper.env, test_helper.starts_no_collision,
            test_helper.goals_no_collision)
        test_helper.assert_path_equality(
            self, test_helper.paths_no_collision, paths)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
