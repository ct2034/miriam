import unittest

import numpy as np
import pytest
from definitions import INVALID

import scenarios.solvers
from scenarios import test_data


class TestSolvers(unittest.TestCase):
    def test_ecbs_success(self):
        # agents that collide in the middle
        starts_success = np.array([
            [0, 0],
            [0, 2]
        ])
        goals_success = np.array([
            [2, 0],
            [2, 2]
        ])
        res = scenarios.solvers.ecbs(
            test_data.env, starts_success, goals_success)
        self.assertTrue(len(res.keys()) != 0)

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
            scenarios.solvers.ecbs(
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
            scenarios.solvers.ecbs(
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
            scenarios.solvers.ecbs(
                test_data.env, starts_deadlocks, goals_deadlocks)
        )
