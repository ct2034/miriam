import unittest

import networkx as nx
from planner.astar_boost.build.libastar_graph import AstarSolver
from planner.astar_boost.converter import initialize_from_graph


class ScenarioStateTest(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:  # type: ignore
        super().__init__(methodName)

    def test_init(self):
        posl = [[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]]
        edges = [[0, 1], [0, 2], [1, 3], [2, 3]]
        a = AstarSolver(posl, edges)
        self.assertAlmostEqual(a.retreive(0).x, 0.1)
        self.assertAlmostEqual(a.retreive(0).y, 0.1)
        self.assertAlmostEqual(a.retreive(1).x, 0.9)
        self.assertAlmostEqual(a.retreive(1).y, 0.1)
        self.assertAlmostEqual(a.retreive(2).x, 0.1)
        self.assertAlmostEqual(a.retreive(2).y, 0.9)
        self.assertAlmostEqual(a.retreive(3).x, 0.9)
        self.assertAlmostEqual(a.retreive(3).y, 0.9)

    def test_plan_simple(self):
        posl = [
            [0.1, 0.1],
            [0.9, 0.1],
            [0.2, 0.8],  # preferred, because closer in
            [0.8, 0.8],  # preferred, because closer in
        ]
        edges = [[0, 1], [1, 2], [2, 3], [0, 3]]
        a = AstarSolver(posl, edges)

        # one path
        path_0_2 = a.plan(0, 2)
        self.assertEqual(len(path_0_2), 3)
        self.assertEqual(path_0_2[0], 0)
        self.assertEqual(path_0_2[1], 3)
        self.assertEqual(path_0_2[2], 2)

        # another path
        path_1_3 = a.plan(1, 3)
        self.assertEqual(len(path_1_3), 3)
        self.assertEqual(path_1_3[0], 1)
        self.assertEqual(path_1_3[1], 2)
        self.assertEqual(path_1_3[2], 3)

    def test_plan_random(self):
        g = nx.random_geometric_graph(10000, 0.02)  # type: nx.Graph
        a = initialize_from_graph(g)
        path = a.plan(0, g.number_of_nodes() - 1)
        self.assertNotEqual(len(path), 0)
