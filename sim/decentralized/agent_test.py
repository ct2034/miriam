#!/usr/bin/env python3

import math
import unittest

import networkx as nx
import numpy as np
from definitions import POS
from sim.decentralized.agent import Agent
from sim.decentralized.policy import Policy, PolicyType, RandomPolicy


class TestAgent(unittest.TestCase):
    def test_init(self):
        env = np.array([[0, 0, 0], [0, 1, 1], [0, 0, 0]])
        a = Agent(env, np.array([0, 2]), PolicyType.RANDOM)
        self.assertEqual(a.pos, 2)
        self.assertIsInstance(a.policy, Policy)
        self.assertIsInstance(a.policy, RandomPolicy)

    def test_hash_equality(self):
        env = np.array([[0, 0, 0], [0, 1, 1], [0, 0, 0]])
        a1 = Agent(env, np.array([0, 2]), PolicyType.RANDOM)
        a2 = Agent(env, np.array([0, 2]), PolicyType.RANDOM)  # same
        self.assertTrue(a1 == a2)

        a3 = Agent(env, np.array([0, 2]), PolicyType.RANDOM)
        self.assertTrue(a1 == a3)
        a3.give_a_goal(np.array([0, 0]))  # setting goal
        self.assertTrue(a1 != a3)  # not same any more

        a4 = Agent(env, np.array([0, 2]), PolicyType.RANDOM)
        a4.give_a_goal(np.array([0, 1]))  # different goal
        self.assertTrue(a1 != a4)
        a4.give_a_goal(np.array([0, 0]))  # same goal
        self.assertTrue(a3 == a4)

        a5 = Agent(env, np.array([0, 1]), PolicyType.RANDOM)  # different pos
        self.assertTrue(a1 != a5)

        # different policy
        a6 = Agent(env, np.array([0, 2]), PolicyType.OPTIMAL)
        self.assertTrue(a1 != a6)

        # in comparison to a4 this switches pos with goal
        a7 = Agent(env, np.array([0, 1]), PolicyType.RANDOM)  # goal of a4
        a7.give_a_goal(np.array([0, 2]))  # pos of a4
        self.assertTrue(a1 != a7)
        self.assertTrue(a4 != a7)

    def test_give_a_goal_and_plan_path(self):
        env = np.array([[0, 0, 0], [0, 1, 1], [0, 0, 0]])
        a = Agent(env, np.array([0, 2]), PolicyType.RANDOM)
        self.assertTrue(a.give_a_goal(np.array([2, 2])))
        p = a.path
        assert p is not None
        self.assertEqual(len(p), 7)
        self.assertEqual(p[1], 1)
        self.assertEqual(p[3], 3)
        self.assertEqual(p[5], 7)

        # if no path can be found
        env = np.array([[0, 0, 0], [0, 1, 1], [0, 1, 0]])
        a = Agent(env, np.array([0, 0]), PolicyType.RANDOM)
        self.assertFalse(a.give_a_goal(np.array([2, 2])))

    def test_is_at_goal(self):
        env = np.array([[0, 0], [0, 0]])
        a = Agent(env, np.array([0, 0]), PolicyType.RANDOM)
        self.assertTrue(a.give_a_goal(np.array([0, 0])))
        self.assertTrue(a.is_at_goal())

        # giving same goal again should return true and still be there
        self.assertTrue(a.give_a_goal(np.array([0, 0])))
        self.assertTrue(a.is_at_goal())

    def test_get_edge_random(self):
        env = np.array([[0, 0], [0, 0]])

        # test random policy
        a = Agent(env, np.array([0, 0]), PolicyType.RANDOM)
        for _ in range(1000):
            node = a.policy.get_edge([a], [0])
            self.assertIn(node, [0, 1, 2])

    def test_what_is_next_step_and_make_next_step(self):
        env = np.array([[0, 0], [0, 1]])
        a = Agent(env, np.array([0, 1]), PolicyType.RANDOM)
        a.give_a_goal(np.array([1, 0]))
        self.assertFalse(a.is_at_goal())

        # what is the next step
        next_step = a.what_is_next_step()
        self.assertTrue(next_step == 0)

        # can we move to the wrong next step
        self.assertRaises(
            AssertionError, lambda: a.make_next_step(99))
        self.assertTrue(a.pos == 1)

        # move to correct next step
        a.make_next_step(next_step)
        self.assertTrue(a.pos == next_step)
        self.assertFalse(a.is_at_goal())

        # what is the next step
        next_step = a.what_is_next_step()
        self.assertTrue(next_step == 2)

        # move final step
        a.make_next_step(next_step)
        self.assertTrue(a.pos == next_step)
        self.assertTrue(a.is_at_goal())

        # doing this step agaion should not change anything
        a.make_next_step(next_step)
        self.assertTrue(a.pos == next_step)
        self.assertTrue(a.is_at_goal())

    def test_same_start_and_goal(self):
        """if agents have same start and goal, the path should only be one long."""
        a = Agent(np.zeros((5, 5)), (2, 3))
        a.give_a_goal((2, 3))
        self.assertTrue(a.is_at_goal())
        assert a.path is not None
        self.assertEqual(len(a.path), 1)

    def test_agent_with_graph(self):
        """Basic operation of agents on graphs"""
        pos_s = {
            0: (0., 0.),
            1: (0., 0.99),
            2: (1., 0.),
            3: (1., 1.)
        }
        g = nx.Graph()
        g.add_edge(0, 1)
        g.add_edge(0, 2)
        g.add_edge(1, 3)
        g.add_edge(2, 3)
        nx.set_node_attributes(g, pos_s, POS)

        a = Agent(g, 0, radius=0.1)
        self.assertTrue(a.give_a_goal(3))

        def general_path_assertions(self):
            self.assertEqual(len(a.path), 3)
            self.assertIsInstance(a.path[0], int)
            self.assertIsInstance(a.path[1], int)
            self.assertIsInstance(a.path[2], int)
            self.assertEqual(a.path[0], 0)
            self.assertEqual(a.path[2], 3)
        general_path_assertions(self)
        self.assertEqual(a.path[1], 1)  # along 1 is quicker


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
