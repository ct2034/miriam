import unittest

import numpy as np
from planner.policylearn.generate_graph import gridmap_to_graph


class GenerateGraphTest(unittest.TestCase):

    def test_gridmap_to_graph(self):
        map_1x2 = np.zeros((1, 2))
        edges_1x2, pos_1x2 = gridmap_to_graph(map_1x2, np.inf, (0, 0))
        self.assertEqual(pos_1x2.shape, (2, 2))
        self.assertEqual(edges_1x2.shape, (2, 1))

        map_3x3 = np.zeros((3, 3))
        edges_3x3, pos_3x3 = gridmap_to_graph(map_3x3, 1, (1, 1))
        self.assertEqual(pos_3x3.shape, (5, 2))
        self.assertEqual(edges_3x3.shape, (2, 4))
