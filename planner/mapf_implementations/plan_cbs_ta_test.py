import unittest

import numpy as np

from planner.mapf_implementations.plan_cbs_ta import plan_in_gridmap


class TestPlanCBS_TA(unittest.TestCase):
    def test_basic_example(self):
        gridmap = np.array([[0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 1]])
        starts = [[1, 2], [3, 2], [2, 1]]
        goals = [[3, 0], [2, 2], [0, 0]]
        res = plan_in_gridmap(gridmap, starts, goals, 10)

        self.assertEqual(res["statistics"]["cost"], 6)
        self.assertEqual(res["statistics"]["makespan"], 3)
        self.assertEqual(res["statistics"]["highLevelExpanded"], 1)
        self.assertEqual(res["statistics"]["lowLevelExpanded"], 9)
        self.assertEqual(res["statistics"]["numTaskAssignments"], 1)

        # agent0 assigened to task 1
        self.assertEqual(res["schedule"]["agent0"][-1]["x"], 2)
        self.assertEqual(res["schedule"]["agent0"][-1]["y"], 2)
        self.assertEqual(res["schedule"]["agent0"][-1]["t"], 1)

        # agent1 assigened to task 0
        self.assertEqual(res["schedule"]["agent1"][-1]["x"], 3)
        self.assertEqual(res["schedule"]["agent1"][-1]["y"], 0)
        self.assertEqual(res["schedule"]["agent1"][-1]["t"], 2)

        # agent2 assigened to task 2
        self.assertEqual(res["schedule"]["agent2"][-1]["x"], 0)
        self.assertEqual(res["schedule"]["agent2"][-1]["y"], 0)
        self.assertEqual(res["schedule"]["agent2"][-1]["t"], 3)
