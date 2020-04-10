#!/usr/bin/env python3

import unittest

import numpy as np

import sim


class TestDecentralizedSim(unittest.TestCase):
    def test_initialize_environment(self):
        env = sim.initialize_environment(10, .5)
        self.assertEqual(env.shape, (10, 10))
        self.assertEqual(np.count_nonzero(env), 50)


if __name__ == "__main__":
    unittest.main()
