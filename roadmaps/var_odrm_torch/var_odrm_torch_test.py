import unittest
from math import sqrt
from random import Random

import torch
from roadmaps.var_odrm_torch.var_odrm_torch import *


class TestVarOdrmTorch(unittest.TestCase):
    def __init__(self, methodName: str) -> None:
        super().__init__(methodName=methodName)
        self.rng = Random(0)
        self.pos = torch.tensor([
            [.2, .2],
            [.2, .8],
            [.8, .8],
            [.7, .3],
        ])
        self.path_nodes = (
            tuple(self.pos[1]),
            tuple(self.pos[0]),
            [1, 0]
        )
        self.path_nodes_plus_one_start = (
            (self.pos[1][0], self.pos[1][1] + 1),
            tuple(self.pos[0]),
            [1, 0]
        )
        self.path_nodes_plus_one_goal = (
            tuple(self.pos[0]),
            (self.pos[1][0], self.pos[1][1] + 1),
            [0, 1]
        )
        self.path_nodes_plus_one_start_goal = (
            (self.pos[1][0], self.pos[1][1] + 1),
            (self.pos[2][0], self.pos[2][1] - 1),
            [1, 2]
        )
        map_img_np = np.full((10, 10), 255)
        self.map_img = tuple(map(tuple, map_img_np))

    def test_sample_points(self):
        n = 100
        points = sample_points(n, self.map_img, self.rng)
        self.assertEqual(n, points.shape[0])
        self.assertEqual(2, points.shape[1])
        self.assertEqual(0, np.count_nonzero(points > 1))
        self.assertEqual(0, np.count_nonzero(points < 0))

    def test_make_graph(self):
        graph = make_graph(self.pos, self.map_img)
        self.assertEqual(5, graph.number_of_edges())
        self.assertEqual(self.pos.shape[0], graph.number_of_nodes())
        self.assertTrue(graph.has_edge(0, 1))
        self.assertTrue(graph.has_edge(0, 3))
        self.assertTrue(graph.has_edge(1, 2))
        self.assertTrue(graph.has_edge(1, 3))
        self.assertTrue(graph.has_edge(2, 3))

    def test_make_paths(self):
        n = self.pos.shape[0]
        graph = make_graph(self.pos, self.map_img)
        paths = make_paths(graph, self.pos, 10, self.rng)
        for path in paths:
            start, goal, node_path = path
            start = np.array(start)
            goal = np.array(goal)
            node_path = np.array(node_path)
            self.assertEqual(0, np.count_nonzero(start > 1))
            self.assertEqual(0, np.count_nonzero(start < 0))
            self.assertEqual(0, np.count_nonzero(goal > 1))
            self.assertEqual(0, np.count_nonzero(goal < 0))
            self.assertGreater(len(node_path), 0)
            self.assertEqual(0, np.count_nonzero(np.array(node_path) < 0))
            self.assertEqual(0, np.count_nonzero(np.array(node_path) > n))

    def test_get_path_len(self):
        self.assertAlmostEqual(
            .6,
            float(get_path_len(self.pos, self.path_nodes)))
        self.assertAlmostEqual(
            6.6,
            float(get_path_len(self.pos, self.path_nodes_plus_one_start)),
            places=5)
        self.assertAlmostEqual(
            6.6,
            float(get_path_len(self.pos, self.path_nodes_plus_one_goal)),
            places=5)
        self.assertAlmostEqual(
            12.6,
            float(get_path_len(self.pos, self.path_nodes_plus_one_start_goal)),
            places=5)

    def test_get_paths_len(self):
        paths = [
            self.path_nodes,
            self.path_nodes_plus_one_start,
            self.path_nodes_plus_one_goal,
            self.path_nodes_plus_one_start_goal
        ]
        self.assertAlmostEqual(
            .6 + 6.6 + 6.6 + 12.6,
            float(get_paths_len(self.pos, paths)),
            places=5)
