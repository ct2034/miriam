#!/usr/bin/env python3

import unittest

import numpy as np
from sim.decentralized.agent import Agent
from sim.decentralized.policy import LearnedPolicy


class TestLearnedPolicy(unittest.TestCase):
    def test_path_until_coll_long(self):
        lp = LearnedPolicy(Agent(np.array([[0, ]]), np.array([0, 0])))

        path = np.array([
            [0, 0],
            [0, 0],
            [1, 1],
            [2, 2],  # <- path_i
            [3, 3],  # coll
            [0, 0],
            [0, 0]
        ])

        path_until_coll = lp._path_until_coll(path, 3, 3)
        assert(max(path_until_coll[0]) == 1)
        assert(max(path_until_coll[1]) == 2)
        assert(max(path_until_coll[2]) == 3)

    def test_path_until_coll_short(self):
        lp = LearnedPolicy(Agent(np.array([[0, ]]), np.array([0, 0])))

        path = np.array([
            [2, 2],  # <- path_i
            [3, 3],  # coll
            [0, 0],
            [0, 0]
        ])

        path_until_coll = lp._path_until_coll(path, 0, 3)
        assert(max(path_until_coll[0]) == 2)
        assert(max(path_until_coll[1]) == 2)
        assert(max(path_until_coll[2]) == 3)

    def test_path_until_coll_shortest(self):
        lp = LearnedPolicy(Agent(np.array([[0, ]]), np.array([0, 0])))

        path = np.array([
            [2, 2]  # <- path_i
        ])

        path_until_coll = lp._path_until_coll(path, 0, 3)
        assert(max(path_until_coll[0]) == 2)
        assert(max(path_until_coll[1]) == 2)
        assert(max(path_until_coll[2]) == 2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
