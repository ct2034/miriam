#!/usr/bin/env python3

import math
import unittest

import networkx as nx
import numpy as np

import agent
import sim
from agent import Agent, Policy


class TestDecentralizedSim(unittest.TestCase):
    def test_init(self):
        env = np.array([[0, 0, 0], [0, 1, 1], [0, 0, 0]])
        a = Agent(env, np.array([0, 2]), Policy.RANDOM)
        self.assertTrue((a.env == env).all())
        self.assertTrue((a.pos == np.array([0, 2])).all())
        self.assertEqual(a.policy, Policy.RANDOM)

    def test_gridmap_to_nx(self):
        env = np.array([[0, 1], [1, 1]])
        a = Agent(env, np.array([0, 0]), Policy.RANDOM)
        a.give_a_goal(np.array([0, 0]))
        self.assertEqual(len(a.env_nx), 1)
        self.assertTrue((0, 0) in a.env_nx)

    def test_give_a_goal_and_plan_path(self):
        env = np.array([[0, 0, 0], [0, 1, 1], [0, 0, 0]])
        a = Agent(env, np.array([0, 2]), Policy.RANDOM)
        self.assertTrue(a.give_a_goal(np.array([2, 2])))
        p = a.path
        self.assertEqual(len(p), 7)
        self.assertTrue((p[1] == [0, 1]).all())
        self.assertTrue((p[3] == [1, 0]).all())
        self.assertTrue((p[5] == [2, 1]).all())

        # if no path can be found
        env = np.array([[0, 0, 0], [0, 1, 1], [0, 1, 0]])
        a = Agent(env, np.array([0, 0]), Policy.RANDOM)
        self.assertFalse(a.give_a_goal(np.array([2, 2])))

    def test_block_edge(self):
        env = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        a = Agent(env, np.array([0, 0]), Policy.RANDOM)
        self.assertTrue(a.give_a_goal(np.array([0, 2])))
        self.assertEqual(len(a.path), 3)  # quick path

        # trying to block a non existant edge
        self.assertRaises(nx.exception.NetworkXError), lambda: a.block_edge(
            (0, 1), (1, 1))

        # blocking edge
        self.assertTrue(a.block_edge((0, 0), (0, 1)))
        self.assertEqual(len(a.path), 7)  # going the long way

        # removing the same edge again should not be a problem
        self.assertTrue(a.block_edge((0, 0), (0, 1)))

        # removing a necessary edge should return false
        self.assertFalse(a.block_edge((2, 0), (2, 1)))

    def test_is_at_goal(self):
        env = np.array([[0, 0], [0, 0]])
        a = Agent(env, np.array([0, 0]), Policy.RANDOM)
        self.assertTrue(a.give_a_goal(np.array([0, 0])))
        self.assertTrue(a.is_at_goal())

        # giving same goal again should return true and still be there
        self.assertTrue(a.give_a_goal(np.array([0, 0])))
        self.assertTrue(a.is_at_goal())

    def test_get_priority_random(self):
        env = np.array([[0, 0], [0, 0]])

        # test random policy
        a = Agent(env, np.array([0, 0]), Policy.RANDOM)
        for _ in range(1000):
            self.assertLessEqual(0, a.get_priority())
            self.assertGreaterEqual(1, a.get_priority())

    def test_get_priority_random_closest(self):
        env = np.array([[0, 0], [0, 0]])

        # with clostest policy
        a = Agent(env, np.array([0, 0]), Policy.CLOSEST)
        self.assertTrue(a.give_a_goal(np.array([1, 0])))
        self.assertAlmostEqual(a.get_priority(), 1.)
        self.assertTrue(a.give_a_goal(np.array([1, 1])))
        self.assertAlmostEqual(a.get_priority(), 1. / math.sqrt(2))

    def test_get_priority_random_fill(self):
        env = np.array([[0, 0, 0, 0, 0],
                        [0, 0, 1, 1, 1],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0]])

        # in the middel
        a = Agent(env, np.array([2, 2]), Policy.FILL)
        self.assertAlmostEqual(a.get_priority(), 3./25.)
        # one left
        a = Agent(env, np.array([2, 1]), Policy.FILL)
        self.assertAlmostEqual(a.get_priority(), 7./25.)
        # low left corner
        a = Agent(env, np.array([4, 0]), Policy.FILL)
        self.assertAlmostEqual(a.get_priority(), 16./25.)
        # top right corner
        a = Agent(env, np.array([0, 4]), Policy.FILL)
        self.assertAlmostEqual(a.get_priority(), 19./25.)

    def test_what_is_next_step_and_make_next_step(self):
        env = np.array([[0, 0], [0, 1]])
        a = Agent(env, np.array([0, 1]), Policy.RANDOM)
        a.give_a_goal(np.array([1, 0]))
        self.assertFalse(a.is_at_goal())

        # what is the next step
        next_step = a.what_is_next_step()
        self.assertTrue(all(next_step == np.array([0, 0])))

        # can we move to the wrong next step
        self.assertRaises(
            AssertionError, lambda: a.make_next_step(np.array([99, 99])))
        self.assertTrue(all(a.pos == np.array([0, 1])))

        # move to correct next step
        a.make_next_step(next_step)
        self.assertTrue(all(a.pos == np.array(next_step)))
        self.assertFalse(a.is_at_goal())

        # what is the next step
        next_step = a.what_is_next_step()
        self.assertTrue(all(next_step == np.array([1, 0])))

        # move final step
        a.make_next_step(next_step)
        self.assertTrue(all(a.pos == np.array(next_step)))
        self.assertTrue(a.is_at_goal())

        # doing this step agaion should not change anything
        a.make_next_step(next_step)
        self.assertTrue(all(a.pos == np.array(next_step)))
        self.assertTrue(a.is_at_goal())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
