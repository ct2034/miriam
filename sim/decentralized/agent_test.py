#!/usr/bin/env python3

import unittest

import numpy as np
import networkx as nx

import sim
from agent import Policy, Agent


class TestDecentralizedSim(unittest.TestCase):
    def test_init(self):
        env = np.array([[0, 0, 0], [0, 1, 1], [0, 0, 0]])
        g = sim.gridmap_to_nx(env)
        a = Agent(env, g, np.array([0, 2]), Policy.RANDOM)
        self.assertTrue((a.env == env).all())
        self.assertEqual(a.env_nx, g)
        self.assertTrue((a.pos == np.array([0, 2])).all())
        self.assertEqual(a.policy, Policy.RANDOM)

    def test_give_a_goal_and_plan_path(self):
        env = np.array([[0, 0, 0], [0, 1, 1], [0, 0, 0]])
        g = sim.gridmap_to_nx(env)
        a = Agent(env, g, np.array([0, 2]), Policy.RANDOM)
        self.assertTrue(a.give_a_goal(np.array([2, 2])))
        p = a.path
        self.assertEqual(len(p), 7)
        self.assertTrue((p[1] == [0, 1]).all())
        self.assertTrue((p[3] == [1, 0]).all())
        self.assertTrue((p[5] == [2, 1]).all())

        # if no path can be found
        env = np.array([[0, 0, 0], [0, 1, 1], [0, 1, 0]])
        g = sim.gridmap_to_nx(env)
        a = Agent(env, g, np.array([0, 0]), Policy.RANDOM)
        self.assertFalse(a.give_a_goal(np.array([2, 2])))

    def test_block_edge(self):
        env = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        g = sim.gridmap_to_nx(env)
        a = Agent(env, g, np.array([0, 0]), Policy.RANDOM)
        self.assertTrue(a.give_a_goal(np.array([0, 2])))
        self.assertEqual(len(a.path), 3)  # quick path

        # trying to block a non existant edge
        self.assertRaises(nx.exception.NetworkXError), lambda: a.block_edge(
            (0, 1), (1, 1))

        # blocking edge
        self.assertTrue(a.block_edge((0, 0), (0, 1)))
        self.assertEqual(len(a.path), 7)  # going the long way

    def test_is_at_goal(self):
        env=np.array([[0, 0], [0, 1]])
        g=sim.gridmap_to_nx(env)
        a=Agent(env, g, np.array([0, 0]), Policy.RANDOM)
        a.give_a_goal(np.array([0, 0]))
        self.assertTrue(a.is_at_goal())

    def test_get_priority(self):
        env=np.array([[0, 0], [0, 1]])
        g=sim.gridmap_to_nx(env)

        # test random policy
        a=Agent(env, g, np.array([0, 0]), Policy.RANDOM)
        for _ in range(1000):
            self.assertLessEqual(0, a.get_priority())
            self.assertGreaterEqual(1, a.get_priority())

        # with clostest policy
        a=Agent(env, g, np.array([0, 0]), Policy.CLOSEST)
        self.assertRaises(NotImplementedError, lambda: a.get_priority())

    def test_what_is_next_step_and_make_next_step(self):
        env=np.array([[0, 0], [0, 1]])
        g=sim.gridmap_to_nx(env)
        a=Agent(env, g, np.array([0, 1]), Policy.RANDOM)
        a.give_a_goal(np.array([1, 0]))
        self.assertFalse(a.is_at_goal())

        # what is the next step
        next_step=a.what_is_next_step()
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
        next_step=a.what_is_next_step()
        self.assertTrue(all(next_step == np.array([1, 0])))

        # move final step
        a.make_next_step(next_step)
        self.assertTrue(all(a.pos == np.array(next_step)))
        self.assertTrue(a.is_at_goal())


if __name__ == "__main__":
    unittest.main()
