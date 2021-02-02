import os
import shutil
import unittest
import uuid

import numpy as np
import pytest
from definitions import INVALID

from scenarios import generators, storage
from scenarios.storage import ResultType


class TestSolvers(unittest.TestCase):
    envvar_str = 'SCENARIO_STORAGE_PATH'
    scenario1 = generators.tracing_pathes_in_the_dark(2, .5, 1, seed=1)
    scenario2 = generators.tracing_pathes_in_the_dark(2, .5, 1, seed=2)
    to_store = 42
    to_store2 = 42.5

    @classmethod
    def setUpClass(cls):
        # making the folder to store data for test in.
        cls.data_path = "/tmp/" + str(uuid.uuid1())
        assert not os.path.exists(cls.data_path)
        os.mkdir(cls.data_path)
        print("folder for testing created under " + cls.data_path)

    @classmethod
    def tearDownClass(cls):
        # remove the folder that the test stored data in.
        shutil.rmtree(cls.data_path)
        print("folder for testing deleted under " + cls.data_path)

    def test_get_filepath(self):
        # without envvar this should fail
        self.assertRaises(AssertionError, lambda: storage.get_filepath((1,)))
        os.environ[self.envvar_str] = TestSolvers.data_path

        # same paths for same scenarios
        path1 = storage.get_filepath(self.scenario1)
        self.assertTrue(path1.startswith(TestSolvers.data_path))
        scenario1_2 = generators.tracing_pathes_in_the_dark(2, .5, 1, seed=1)
        path1_2 = storage.get_filepath(scenario1_2)
        self.assertTrue(path1_2.startswith(TestSolvers.data_path))
        self.assertEqual(path1, path1_2)

        # different paths for different scenarios
        path2 = storage.get_filepath(self.scenario2)
        self.assertTrue(path2.startswith(TestSolvers.data_path))
        self.assertNotEqual(path1, path2)

    # test has_file
        # has no been created yet
        self.assertFalse(storage.has_file(self.scenario1))

        # then it should have
        storage.save_result(
            self.scenario1, ResultType.ECBS_PATHS, self.to_store)
        self.assertTrue(storage.has_file(self.scenario1))

    # test_has_result
        storage.save_result(
            self.scenario1, ResultType.INDEP_PATHS, self.to_store2)

        # should not have this
        self.assertFalse(storage.has_result(
            self.scenario1, ResultType.ECBS_DATA))

        # should have (was created before)
        self.assertTrue(storage.has_result(
            self.scenario1, ResultType.INDEP_PATHS))

    # test_get_result
        storage.save_result(
            self.scenario1, ResultType.ICTS_PATHS, self.to_store2)

        # should not have this result type
        self.assertRaises(AssertionError, lambda: storage.get_result(
            self.scenario1, ResultType.ICTS_INFO))

        # should not have this file
        self.assertRaises(AssertionError, lambda: storage.get_result(
            self.scenario2, ResultType.ICTS_PATHS))

        # should have (was created before)
        self.assertEqual(self.to_store2, storage.get_result(
            self.scenario1, ResultType.ICTS_PATHS))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
