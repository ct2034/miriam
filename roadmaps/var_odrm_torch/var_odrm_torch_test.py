import unittest
from random import Random

import numpy as np
import pytest
import torch

from roadmaps.var_odrm_torch.var_odrm_torch import (
    get_path_len,
    get_paths_len,
    make_graph_and_flann,
    make_paths,
    sample_points,
)


class TestVarOdrmTorch(unittest.TestCase):
    def __init__(self, methodName: str) -> None:
        super().__init__(methodName=methodName)
        self.rng = Random(0)
        self.pos = torch.tensor(
            [
                [0.2, 0.2],
                [0.2, 0.8],
                [0.8, 0.8],
                [0.7, 0.3],
            ]
        )
        self.path_nodes = (tuple(self.pos[1]), tuple(self.pos[0]), [1, 0])
        self.path_nodes_plus_one_start = (
            (self.pos[1][0], self.pos[1][1] + 1),
            tuple(self.pos[0]),
            [1, 0],
        )
        self.path_nodes_plus_one_goal = (
            tuple(self.pos[0]),
            (self.pos[1][0], self.pos[1][1] + 1),
            [0, 1],
        )
        self.path_nodes_plus_one_start_goal = (
            (self.pos[1][0], self.pos[1][1] + 1),
            (self.pos[2][0], self.pos[2][1] - 1),
            [1, 2],
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

    @pytest.mark.skip(reason="Not sure how to fix this, TODO")
    def test_make_graph(self):
        graph, _ = make_graph_and_flann(self.pos, self.map_img, 4, rng=Random(0))
        # 5 delaunay-edges and 4 self-edges
        self.assertEqual(5 + 4, graph.number_of_edges())
        self.assertEqual(self.pos.shape[0], graph.number_of_nodes())
        self.assertTrue(graph.has_edge(0, 1))
        self.assertTrue(graph.has_edge(0, 3))
        self.assertTrue(graph.has_edge(1, 2))
        self.assertTrue(graph.has_edge(1, 3))
        self.assertTrue(graph.has_edge(2, 3))

    @pytest.mark.skip(reason="Not sure how to fix this, TODO")
    def test_make_paths(self):
        n_nodes = self.pos.shape[0]
        graph, _ = make_graph_and_flann(self.pos, self.map_img, n_nodes, rng=Random(0))
        paths = make_paths(graph, 10, self.map_img, self.rng)
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
            self.assertEqual(0, np.count_nonzero(np.array(node_path) > n_nodes))

    def test_get_path_len_training(self):
        self.assertAlmostEqual(
            0.6, float(get_path_len(self.pos, self.path_nodes, True))
        )
        self.assertAlmostEqual(
            8.6,
            float(get_path_len(self.pos, self.path_nodes_plus_one_start, True)),
            places=5,
        )
        self.assertAlmostEqual(
            8.6,
            float(get_path_len(self.pos, self.path_nodes_plus_one_goal, True)),
            places=5,
        )
        self.assertAlmostEqual(
            16.6,
            float(get_path_len(self.pos, self.path_nodes_plus_one_start_goal, True)),
            places=5,
        )

    def test_get_path_len_testing(self):
        self.assertAlmostEqual(
            0.6, float(get_path_len(self.pos, self.path_nodes, False))
        )
        self.assertAlmostEqual(
            1.6,
            float(get_path_len(self.pos, self.path_nodes_plus_one_start, False)),
            places=5,
        )
        self.assertAlmostEqual(
            1.6,
            float(get_path_len(self.pos, self.path_nodes_plus_one_goal, False)),
            places=5,
        )
        self.assertAlmostEqual(
            2.6,
            float(get_path_len(self.pos, self.path_nodes_plus_one_start_goal, False)),
            places=5,
        )

    def test_get_paths_len(self):
        paths = [
            self.path_nodes,
            self.path_nodes_plus_one_start,
            self.path_nodes_plus_one_goal,
            self.path_nodes_plus_one_start_goal,
        ]
        self.assertAlmostEqual(
            0.6 + 8.6 + 8.6 + 16.6,
            float(get_paths_len(self.pos, paths, True)),
            places=5,
        )
        self.assertAlmostEqual(
            0.6 + 1.6 + 1.6 + 2.6,
            float(get_paths_len(self.pos, paths, False)),
            places=5,
        )


if __name__ == "__main__":
    pytest.main(["-v", __file__])
