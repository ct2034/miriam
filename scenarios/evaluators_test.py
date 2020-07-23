import unittest

import numpy as np

import scenarios.evaluators


class TestEvaluators(unittest.TestCase):
    env = np.array([
        [0, 0, 0],
        [1, 0, 1],
        [0, 0, 0]
    ])

    def test_is_well_fromed(self):
        starts_wf = np.array([
            [0, 0],
            [0, 2]
        ])
        goals_wf = np.array([
            [2, 0],
            [2, 2]
        ])
        self.assertTrue(
            scenarios.evaluators.is_well_formed(
                self.env, starts_wf, goals_wf, ignore_cache=True)
        )
        starts_nwf = np.array([
            [1, 1],
            [0, 2]
        ])
        goals_nwf = np.array([
            [2, 0],
            [2, 2]
        ])
        self.assertFalse(
            scenarios.evaluators.is_well_formed(
                self.env, starts_nwf, goals_nwf, ignore_cache=True)
        )

    def test_ecbs_cost(self):
        # agents that collide in the middle
        starts_4_5 = np.array([
            [0, 0],
            [0, 2]
        ])
        goals_4_5 = np.array([
            [2, 0],
            [2, 2]
        ])
        self.assertAlmostEqual(
            4.5, scenarios.evaluators.cost_ecbs(
                self.env, starts_4_5, goals_4_5, ignore_cache=True)
        )
        # agents that don't collide
        starts_2 = np.array([
            [0, 0],
            [2, 0]
        ])
        goals_2 = np.array([
            [0, 2],
            [2, 2]
        ])
        self.assertAlmostEqual(
            2, scenarios.evaluators.cost_ecbs(
                self.env, starts_2, goals_2, ignore_cache=True)
        )
        # one agents path is not possible
        starts_invalid = np.array([
            [0, 0],
            [1, 0]  # obstacle
        ])
        goals_invalid = np.array([
            [0, 2],
            [1, 2]  # obstacle
        ])
        self.assertAlmostEqual(
            scenarios.evaluators.INVALID,
            scenarios.evaluators.cost_ecbs(
                self.env, starts_invalid, goals_invalid, ignore_cache=True)
        )
        # trying to make is time out ...
        starts_timeout = np.array([
            [0, 0],
            [0, 1],
            [0, 2],
            [1, 1],
            [2, 0],
            [2, 1],
            [2, 2]
        ])
        goals_timeout = np.array([
            [2, 0],
            [2, 1],
            [2, 2],
            [0, 0],
            [0, 1],
            [0, 2],
            [1, 1]
        ])
        self.assertAlmostEqual(
            scenarios.evaluators.INVALID,
            scenarios.evaluators.cost_ecbs(
                self.env, starts_timeout, goals_timeout, ignore_cache=True)
        )

    def test_cost_independant(self):
        # agents that would collide in the middle
        starts_4 = np.array([
            [0, 0],
            [0, 2]
        ])
        goals_4 = np.array([
            [2, 0],
            [2, 2]
        ])
        self.assertAlmostEqual(
            4, scenarios.evaluators.cost_independant(
                self.env, starts_4, goals_4, ignore_cache=True)
        )
        # agents that don't collide
        starts_2 = np.array([
            [0, 0],
            [2, 0]
        ])
        goals_2 = np.array([
            [0, 2],
            [2, 2]
        ])
        self.assertAlmostEqual(
            2, scenarios.evaluators.cost_independant(
                self.env, starts_2, goals_2, ignore_cache=True)
        )
        # one agents path is not possible
        starts_invalid = np.array([
            [0, 0],
            [1, 0]  # obstacle
        ])
        goals_invalid = np.array([
            [0, 2],
            [2, 2]
        ])
        self.assertAlmostEqual(
            scenarios.evaluators.INVALID,
            scenarios.evaluators.cost_independant(
                self.env, starts_invalid, goals_invalid, ignore_cache=True)
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
