import unittest

import networkx as nx
import numpy as np

import tools


class TestTools(unittest.TestCase):
    def test_hasher(self):
        def make_base():
            return {"np": np.zeros([1000, 1000], dtype=int), "int": 42}

        params_base = make_base()

        params_same_a = make_base()
        params_same_a["np"][42, 43] = 1
        params_same_b = make_base()
        params_same_b["np"][42, 43] = 1

        self.assertEqual(
            tools.hasher((), params_same_a), tools.hasher((), params_same_b)
        )
        self.assertEqual(
            tools.hasher([params_same_a["np"], params_same_b["np"]], {}),
            tools.hasher([params_same_b["np"], params_same_a["np"]], {}),
        )

        params_diff_np_a = make_base()
        params_diff_np_a["np"][42, 43] = 1
        params_diff_np_b = make_base()
        params_diff_np_b["np"][42, 431] = 1

        self.assertNotEqual(
            tools.hasher((), params_diff_np_a), tools.hasher((), params_diff_np_b)
        )
        self.assertNotEqual(
            tools.hasher((), params_base), tools.hasher((), params_diff_np_a)
        )
        self.assertNotEqual(
            tools.hasher((), params_base), tools.hasher((), params_diff_np_b)
        )

        self.assertNotEqual(
            tools.hasher([params_diff_np_a["np"]], {}),
            tools.hasher([params_diff_np_b["np"]], {}),
        )
        self.assertNotEqual(
            tools.hasher([params_diff_np_a["np"], params_diff_np_b["np"]], {}),
            tools.hasher([params_diff_np_b["np"], params_diff_np_a["np"]], {}),
        )
        self.assertNotEqual(
            tools.hasher([params_base["np"]], {}),
            tools.hasher([params_diff_np_a["np"]], {}),
        )
        self.assertNotEqual(
            tools.hasher([params_base["np"]], {}),
            tools.hasher([params_diff_np_b["np"]], {}),
        )

        params_diff_int_a = make_base()
        params_diff_int_a["int"] = 43
        params_diff_int_b = make_base()
        params_diff_int_b["int"] = 44

        self.assertNotEqual(
            tools.hasher((), params_diff_int_a), tools.hasher((), params_diff_int_b)
        )
        self.assertNotEqual(
            tools.hasher((), params_base), tools.hasher((), params_diff_int_a)
        )
        self.assertNotEqual(
            tools.hasher((), params_base), tools.hasher((), params_diff_int_b)
        )

    def test_graphs(self):
        g1 = nx.Graph([(0, 1)])
        nx.set_node_attributes(g1, {0: {"pos": [0, 0]}})

        g2 = nx.Graph([(0, 1)])
        nx.set_node_attributes(g2, {0: {"pos": [0, 0.1]}})
        self.assertNotEqual(tools.hasher([g1], {}), tools.hasher([g2], {}))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
