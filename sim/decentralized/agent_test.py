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
        self.assertTrue((a.env == env).all())
        self.assertTrue((a.pos == np.array([0, 2])).all())
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

        a6 = Agent(env, np.array([0, 2]), PolicyType.FILL)  # different policy
        self.assertTrue(a1 != a6)

        # in comparison to a4 this switches pos with goal
        a7 = Agent(env, np.array([0, 1]), PolicyType.RANDOM)  # goal of a4
        a7.give_a_goal(np.array([0, 2]))  # pos of a4
        self.assertTrue(a1 != a7)
        self.assertTrue(a4 != a7)

    def test_gridmap_to_nx(self):
        env = np.array([[0, 0], [1, 1]])
        a = Agent(env, np.array([0, 0]))
        a.give_a_goal(np.array([0, 0]))

        self.assertEqual(len(a.env_nx), 24)  # 4 * 6
        for i in range(4):
            self.assertTrue((0, 0, i) in a.env_nx)
            self.assertTrue((0, 1, i) in a.env_nx)

        self.assertEqual(len(a.env_nx.edges), 44)  # 4 * 6 * 4
        for i in range(3):
            self.assertTrue(((0, 0, i), (0, 0, i+1)) in a.env_nx.edges)
            self.assertTrue(((0, 1, i), (0, 1, i+1)) in a.env_nx.edges)
            self.assertTrue(((0, 0, i), (0, 1, i+1)) in a.env_nx.edges)
            self.assertTrue(((0, 1, i), (0, 0, i+1)) in a.env_nx.edges)

    def test_give_a_goal_and_plan_path(self):
        env = np.array([[0, 0, 0], [0, 1, 1], [0, 0, 0]])
        a = Agent(env, np.array([0, 2]), PolicyType.RANDOM)
        self.assertTrue(a.give_a_goal(np.array([2, 2])))
        p = a.path
        self.assertEqual(len(p), 7)
        self.assertTrue(p[1] == (0, 1, 1))
        self.assertTrue(p[3] == (1, 0, 3))
        self.assertTrue(p[5] == (2, 1, 5))

        # if no path can be found
        env = np.array([[0, 0, 0], [0, 1, 1], [0, 1, 0]])
        a = Agent(env, np.array([0, 0]), PolicyType.RANDOM)
        self.assertFalse(a.give_a_goal(np.array([2, 2])))

    def test_block_edge(self):
        env = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        edge_to_block = ((0, 0), (0, 1), 0)
        a = Agent(env, np.array([0, 0]), PolicyType.RANDOM)
        self.assertTrue(a.give_a_goal((0, 2)))
        self.assertEqual(len(a.path), 3)  # quick path

        # trying to block a non existant edge
        self.assertRaises(nx.exception.NetworkXError), lambda: a.block_edge(
            (0, 1), (1, 1), 0)

        # blocking it, path should still be possible
        self.assertTrue(a.block_edge(edge_to_block))
        self.assertEqual(len(a.path), 4)  # waiting

        # removing the same edge again should not be a problem
        self.assertTrue(a.block_edge(edge_to_block))

        # and we should also be able to give another goal
        self.assertTrue(a.give_a_goal(np.array([1, 0])))
        self.assertEqual(len(a.path), 2)

    def test_block_node(self):
        env = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
        node_to_block = (1, 1, 2)
        a = Agent(env, np.array([0, 0]), PolicyType.RANDOM)
        self.assertTrue(a.give_a_goal(np.array([2, 2])))
        self.assertEqual(len(a.path), 5)  # quick path

        # trying to block a non existant node
        self.assertRaises(nx.exception.NetworkXError), lambda: a.block_node(
            (0, 1), (0, 2), 0)

        # blocking it, path should still be possible
        self.assertTrue(a.block_node(node_to_block))
        self.assertEqual(len(a.path), 6)  # waiting

        # blocking the same node again should not be a problem
        self.assertTrue(a.block_node(node_to_block))

        # and we should also be able to give another goal
        self.assertTrue(a.give_a_goal(np.array([1, 2])))
        self.assertEqual(len(a.path), 5)

        # removing block
        a.blocked_nodes = set()

        # and we should also be able to give another goal
        self.assertTrue(a.give_a_goal(np.array([1, 2])))
        self.assertEqual(len(a.path), 4)

    def test_is_at_goal(self):
        env = np.array([[0, 0], [0, 0]])
        a = Agent(env, np.array([0, 0]), PolicyType.RANDOM)
        self.assertTrue(a.give_a_goal(np.array([0, 0])))
        self.assertTrue(a.is_at_goal())

        # giving same goal again should return true and still be there
        self.assertTrue(a.give_a_goal(np.array([0, 0])))
        self.assertTrue(a.is_at_goal())

    def test_get_priority_random(self):
        env = np.array([[0, 0], [0, 0]])

        # test random policy
        a = Agent(env, np.array([0, 0]), PolicyType.RANDOM)
        for _ in range(1000):
            self.assertLessEqual(0, a.get_priority(0))
            self.assertGreaterEqual(1, a.get_priority(0))

    def test_get_priority_random_closest(self):
        env = np.array([[0, 0], [0, 0]])

        # with clostest policy
        a = Agent(env, np.array([0, 0]), PolicyType.CLOSEST)
        self.assertTrue(a.give_a_goal(np.array([1, 0])))
        self.assertAlmostEqual(a.get_priority(0), 1.)
        self.assertTrue(a.give_a_goal(np.array([1, 1])))
        self.assertAlmostEqual(a.get_priority(0), 1. / math.sqrt(2))

    def test_get_priority_random_fill(self):
        env = np.array([[0, 0, 0, 0, 0],
                        [0, 0, 1, 1, 1],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0]])

        # in the middel
        a = Agent(env, np.array([2, 2]), PolicyType.FILL)
        self.assertAlmostEqual(a.get_priority(0), 3./25.)
        # one left
        a = Agent(env, np.array([2, 1]), PolicyType.FILL)
        self.assertAlmostEqual(a.get_priority(0), 7./25.)
        # low left corner
        a = Agent(env, np.array([4, 0]), PolicyType.FILL)
        self.assertAlmostEqual(a.get_priority(0), 16./25.)
        # top right corner
        a = Agent(env, np.array([0, 4]), PolicyType.FILL)
        self.assertAlmostEqual(a.get_priority(0), 19./25.)

    def test_what_is_next_step_and_make_next_step(self):
        env = np.array([[0, 0], [0, 1]])
        a = Agent(env, np.array([0, 1]), PolicyType.RANDOM)
        a.give_a_goal(np.array([1, 0]))
        self.assertFalse(a.is_at_goal())

        # what is the next step
        next_step = a.what_is_next_step()
        self.assertTrue(all(next_step[:2] == np.array([0, 0])))

        # can we move to the wrong next step
        self.assertRaises(
            AssertionError, lambda: a.make_next_step((99, 99)))
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

    def test_same_start_and_goal(self):
        """if agents have same start and goal, the path should only be one long."""
        a = Agent(np.zeros((5, 5)), (2, 3))
        a.give_a_goal((2, 3))
        self.assertTrue(a.is_at_goal())
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

        a = Agent(g, 0)
        self.assertTrue(a.give_a_goal(3))

        def general_path_assertions(self):
            self.assertEqual(len(a.path), 3)
            self.assertEqual(len(a.path[0]), 2)
            self.assertEqual(a.path[0], (0, 0))
            self.assertEqual(a.path[2], (3, 2))
        general_path_assertions(self)
        self.assertEqual(a.path[1], (1, 1))  # along 1 is quicker

        # if we block that node ..
        self.assertTrue(a.block_node((1, 1)))
        general_path_assertions(self)
        self.assertEqual(a.path[1], (2, 1))  # along 2

        # unblock
        a.blocked_nodes = set()
        self.assertTrue(a.give_a_goal(3))
        general_path_assertions(self)
        self.assertEqual(a.path[1], (1, 1))  # along 1 is quicker

        # now blocking an edge
        self.assertTrue(a.block_edge(((1, 1), (3, 2))))
        general_path_assertions(self)
        self.assertEqual(a.path[1], (2, 1))  # along 2


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
