#!/usr/bin/env python3

import unittest
from unittest.mock import MagicMock

import numpy as np
from sim.decentralized.agent import Agent
from sim.decentralized.policy import LearnedPolicy, PolicyType
from sim.decentralized.runner import run_a_scenario


class TestLearnedPolicy(unittest.TestCase):
    def test_learned_path_until_coll_long(self):
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

    def test_learned_path_until_coll_short(self):
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

    def test_learned_path_until_coll_shortest(self):
        lp = LearnedPolicy(Agent(np.array([[0, ]]), np.array([0, 0])))

        path = np.array([
            [2, 2]  # <- path_i
        ])

        path_until_coll = lp._path_until_coll(path, 0, 3)
        assert(max(path_until_coll[0]) == 2)
        assert(max(path_until_coll[1]) == 2)
        assert(max(path_until_coll[2]) == 2)

    def test_learned_get_priority_calling_model(self):
        predictMock = MagicMock(return_value=[[.5, ], ])
        env = np.zeros((3, 3))
        env[1, [0, 2]] = 1
        a1 = Agent(env, np.array([0, 0]), PolicyType.LEARNED)
        a1.give_a_goal(np.array([2, 2]))
        a1.policy.model.predict = predictMock
        a2 = Agent(env, np.array([0, 2]), PolicyType.LEARNED)
        a2.give_a_goal(np.array([2, 0]))
        a2.policy.model.predict = predictMock
        _, _, _, _, success = run_a_scenario(env, [a1, a2], False)
        self.assertEqual(success, 1)

        def  hist(x): return np.histogram(x, bins=100, range=(0, 1))[0]
        predictMock.assert_called()
        for call in predictMock.call_args_list:
            model_data = call.args[0].numpy()
            self.assertEqual(model_data.shape, (1, 7, 7, 3, 5))
            for t in range(3):
                # layer 0 "map" ...............................................
                self.assertEqual(model_data[0, 3, 3, t, 0], 0)

                # layer 1 "poses" .............................................
                hist1 = hist(model_data[0, :, :, t, 1])
                self.assertEqual(np.argmax(hist1), 0)  # majority black
                if t == 2:
                    self.assertEqual(model_data[0, 3, 3, t, 1], 1)

                # layer 2 "own path" ..........................................
                hist2 = hist(model_data[0, :, :, t, 2])
                second_highest_value = hist2[hist2.argsort()[-2]]
                # only one path with one shade each
                self.assertEqual(second_highest_value, 1)
                # middle always on path
                self.assertLess(model_data[0, 3, 3, t, 2], 1)
                self.assertGreater(model_data[0, 3, 3, t, 2], 0)

                # layer 3 "opponent path" .....................................
                hist3 = hist(model_data[0, :, :, t, 3])
                second_highest_value = hist3[hist3.argsort()[-2]]
                # only one path with one shade each
                self.assertEqual(second_highest_value, 1)
                if t == 2:
                    # middle on path
                    self.assertLess(model_data[0, 3, 3, t, 3], 1)
                    self.assertGreater(model_data[0, 3, 3, t, 3], 0)

                # layer 4 "all paths" .........................................
                hist4 = hist(model_data[0, :, :, t, 4])
                # majority black    tbd: probably not with many many agents
                self.assertEqual(np.argmax(hist4), 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
