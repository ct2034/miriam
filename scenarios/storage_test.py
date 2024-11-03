import os
import random
import unittest

import numpy as np

from definitions import INVALID
from scenarios import generators, storage, test_helper
from scenarios.storage import ResultType, to_key_string


class TestStorage(unittest.TestCase):
    scenario1 = generators.tracing_paths_in_the_dark(2, 0.5, 1, rng=random.Random(1))
    scenario2 = generators.tracing_paths_in_the_dark(2, 0.5, 1, rng=random.Random(2))
    to_store = 42
    to_store2 = 42.5
    to_store3 = 42.6

    @classmethod
    def setUpClass(cls):
        test_helper.unset_envvar()
        # making the folder to store data for test in.
        cls.data_path = test_helper.make_cache_folder_and_set_envvar(set_envvar=False)

    @classmethod
    def tearDownClass(cls):
        # remove the folder that the test stored data in.
        test_helper.remove_cache_folder_and_unset_envvar()

    def test_storage(self):
        # without envvar this should fail
        # test_helper.unset_envvar()
        # self.assertRaises(AssertionError, lambda: storage.get_filepath((1, 1, 1)))
        os.environ[test_helper.ENVVAR_STORAGE_PATH_STR] = TestStorage.data_path

        # same paths for same scenarios
        path1 = storage.get_filepath(self.scenario1)
        self.assertTrue(path1.startswith(TestStorage.data_path))
        scenario1_2 = generators.tracing_paths_in_the_dark(
            2, 0.5, 1, rng=random.Random(1)
        )
        path1_2 = storage.get_filepath(scenario1_2)
        self.assertTrue(path1_2.startswith(TestStorage.data_path))
        self.assertEqual(path1, path1_2)

        # different paths for different scenarios
        path2 = storage.get_filepath(self.scenario2)
        self.assertTrue(path2.startswith(TestStorage.data_path))
        self.assertNotEqual(path1, path2)

        # test has_file .......................................................
        # has no been created yet
        self.assertFalse(storage.has_file(self.scenario1))

        # then it should have
        storage.save_result(self.scenario1, ResultType.ECBS_PATHS, {}, self.to_store)
        self.assertTrue(storage.has_file(self.scenario1))

        # test_has_result .....................................................
        storage.save_result(self.scenario1, ResultType.INDEP_PATHS, {}, self.to_store2)

        # should not have this
        self.assertFalse(storage.has_result(self.scenario1, ResultType.ECBS_DATA, {}))

        # should have (was created before)
        self.assertTrue(storage.has_result(self.scenario1, ResultType.INDEP_PATHS, {}))

        # test_get_result .....................................................
        storage.save_result(self.scenario1, ResultType.ICTS_PATHS, {}, self.to_store2)

        # should not have this result type
        self.assertRaises(
            AssertionError,
            lambda: storage.get_result(self.scenario1, ResultType.ICTS_INFO, {}),
        )

        # should not have this file
        self.assertRaises(
            AssertionError,
            lambda: storage.get_result(self.scenario2, ResultType.ICTS_PATHS, {}),
        )

        # should have (was created before)
        self.assertEqual(
            self.to_store2,
            storage.get_result(self.scenario1, ResultType.ICTS_PATHS, {}),
        )

    def test_storage_solver_params(self):
        dummy_solver_params_a = {"a": 1}
        dummy_solver_params_b = {"b": 1}
        # precheck
        self.assertFalse(
            storage.has_result(
                self.scenario1, ResultType.ECBS_PATHS, dummy_solver_params_a
            )
        )
        self.assertFalse(
            storage.has_result(
                self.scenario1, ResultType.ECBS_PATHS, dummy_solver_params_b
            )
        )

        # store
        self.assertFalse(
            storage.save_result(
                self.scenario1,
                ResultType.ECBS_PATHS,
                dummy_solver_params_a,
                self.to_store3,
            )
        )

        # check again
        self.assertTrue(
            storage.has_result(
                self.scenario1, ResultType.ECBS_PATHS, dummy_solver_params_a
            )
        )
        self.assertFalse(
            storage.has_result(
                self.scenario1, ResultType.ECBS_PATHS, dummy_solver_params_b
            )
        )

        # retreive
        self.assertEqual(
            self.to_store3,
            storage.get_result(
                self.scenario1, ResultType.ECBS_PATHS, dummy_solver_params_a
            ),
        )

    def test_to_key_string(self):
        s = to_key_string(ResultType.ECBS_DATA, {"c": 4, "b": 33, "a": 1, "x": True})
        self.assertEqual(s, "ECBS_DATA_a=1_b=33_c=4_x=True")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
