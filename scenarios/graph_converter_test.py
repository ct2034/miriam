from scenarios.graph_converter import *
import unittest
import random
from definitions import OBSTACLE


class TestGraphConverter(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName=methodName)
        self.env = np.zeros((8, 8))

    def test_gridmap_to_nx(self):
        small_env = np.zeros((3, 3))
        small_env[0, 0] = OBSTACLE
        small_env[0, 1] = OBSTACLE
        small_env[1, 0] = OBSTACLE
        small_env[1, 1] = OBSTACLE
        g = gridmap_to_nx(small_env)
        self.assertEqual(len(g.nodes), 5)
        self.assertEqual(len(g.edges), 4)

    def test_coordinate_to_node(self):
        self.assertEqual(coordinate_to_node(self.env, (0, 0)), 0)
        self.assertEqual(coordinate_to_node(self.env, (0, 7)), 7)
        self.assertEqual(coordinate_to_node(self.env, (7, 0)), 56)
        self.assertEqual(coordinate_to_node(self.env, (7, 7)), 63)

    def test_node_to_coordinate(self):
        self.assertEqual(node_to_coordinate(self.env, 0), (0, 0))
        self.assertEqual(node_to_coordinate(self.env, 7), (0, 7))
        self.assertEqual(node_to_coordinate(self.env, 56), (7, 0))
        self.assertEqual(node_to_coordinate(self.env, 63), (7, 7))

    def test_node_to_coordinate_fail(self):
        with self.assertRaises(IndexError):
            node_to_coordinate(self.env, 64)

    def test_coordinate_to_node_fail(self):
        with self.assertRaises(IndexError):
            coordinate_to_node(self.env, (8, 0))

    def test_two_way(self):
        rng = random.Random(0)
        for _ in range(10):
            node = rng.randint(0, 63)
            self.assertEqual(coordinate_to_node(
                self.env, node_to_coordinate(self.env, node)), node)
