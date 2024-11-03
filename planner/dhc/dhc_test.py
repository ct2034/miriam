import unittest
from random import Random

import numpy as np

from definitions import INVALID
from planner.dhc.eval import eval
from scenarios.generators import tracing_paths_in_the_dark


class DhcTest(unittest.TestCase):
    def test_eval(self):
        rng = Random(0)
        size = 10
        fill = 0.5
        n_agents = 5
        env, starts, goals = tracing_paths_in_the_dark(size, fill, n_agents, rng)
        res = eval(env, starts, goals)
        success, steps, paths = res
        self.assertTrue(success)
        self.assertGreater(steps, 0)
        self.assertEqual(len(paths), n_agents)
        for i_a in range(n_agents):
            self.assertEqual(len(paths[i_a]), steps + 1)
            self.assertEqual(tuple(paths[i_a][0]), tuple(starts[i_a]))
            self.assertEqual(tuple(paths[i_a][-1]), tuple(goals[i_a]))

    def test_eval_invalid_tight(self):
        env_invalid = np.array([[0, 0], [1, 1]])  # square
        starts_invalid = np.array([[0, 0], [0, 1]])
        goals_invalid = np.array([[0, 1], [0, 0]])  # swap

        res = eval(env_invalid, starts_invalid, goals_invalid)
        self.assertEqual(res, INVALID)

    def test_eval_invalid_obstacle(self):
        env_invalid = np.array([[0, 0], [1, 1]])  # square
        starts_invalid = np.array([[0, 0], [0, 1]])
        # impossible to reach on map
        goals_invalid = np.array([[1, 0], [1, 1]])

        res = eval(env_invalid, starts_invalid, goals_invalid)
        self.assertEqual(res, INVALID)
