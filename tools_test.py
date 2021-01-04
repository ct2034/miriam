import tools

import numpy as np
import unittest


class TestTools(unittest.TestCase):
    def test_hasher(self):
        def make_base(): return {"np": np.zeros(
            [100, 100], dtype=np.int8), "int": 42}
        params_base = make_base()

        params_same_a = make_base()
        params_same_a["np"][42, 43] = 99
        params_same_b = make_base()
        params_same_b["np"][42, 43] = 99

        self.assertEqual(tools.hasher((), params_same_a),
                         tools.hasher((), params_same_b))

        params_diff_np_a = make_base()
        params_diff_np_a["np"][42, 43] = 98
        params_diff_np_b = make_base()
        params_diff_np_b["np"][42, 43] = 97

        self.assertNotEqual(tools.hasher((), params_diff_np_a),
                            tools.hasher((), params_diff_np_b))
        self.assertNotEqual(tools.hasher((), params_base),
                            tools.hasher((), params_diff_np_a))
        self.assertNotEqual(tools.hasher((), params_base),
                            tools.hasher((), params_diff_np_b))

        params_diff_int_a = make_base()
        params_diff_int_a["int"] = 43
        params_diff_int_b = make_base()
        params_diff_int_b["int"] = 44

        self.assertNotEqual(tools.hasher((), params_diff_int_a),
                            tools.hasher((), params_diff_int_b))
        self.assertNotEqual(tools.hasher((), params_base),
                            tools.hasher((), params_diff_int_a))
        self.assertNotEqual(tools.hasher((), params_base),
                            tools.hasher((), params_diff_int_b))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
