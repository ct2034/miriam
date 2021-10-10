import unittest

import numpy as np
from definitions import INVALID
from planner.mapf_implementations.plan_ecbs import plan_in_gridmap
from scenarios import test_helper


class PlanEcbsTest(unittest.TestCase):
    def test_solvable_only_if_dissappear(self):
        res = plan_in_gridmap(
            test_helper.env_deadlock,
            starts=[
                (1, 0),
                (1, 2)
            ],
            goals=[
                (1, 2),
                (1, 1)
            ],
            suboptimality=1.5,
            timeout=10,
            disappear_at_goal=True  # can be solved 
        )
        self.assertNotEqual(res, INVALID)

    def test_solvable_only_if_dissappear_invalid(self):
        res = plan_in_gridmap(
            test_helper.env_deadlock,
            starts=[
                (1, 0),
                (1, 2)
            ],
            goals=[
                (1, 2),
                (1, 1)
            ],
            suboptimality=1.5,
            timeout=10,
            disappear_at_goal=False  # can not be solved then
        )
        self.assertEqual(res, INVALID)
