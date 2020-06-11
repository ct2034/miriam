#!/usr/bin/env python3

import unittest

import numpy as np

import sim
from agent import Policy, Agent


class TestDecentralizedSim(unittest.TestCase):    
    
    def test_plan_path(self):
        env = np.array([[0, 0, 0], [0, 1, 1], [0, 0, 0]])
        g = sim.gridmap_to_nx(env)
        a = Agent(env, g, np.array([0, 2]), Policy.RANDOM)
        a.give_a_goal(np.array([2, 2]))
        p = a.path
        self.assertEqual(len(p), 7)
        self.assertTrue((p[1] == [0, 1]).all())
        self.assertTrue((p[3] == [1, 0]).all())
        self.assertTrue((p[5] == [2, 1]).all())