import os
import shutil
import unittest
import uuid

import numpy as np
from definitions import INVALID

from scenarios import generators, storage
from scenarios.storage import ResultType


class TestSolvers(unittest.TestCase):
    envvar_str = 'SCENARIO_STORAGE_PATH'
    scenario1 = generators.tracing_pathes_in_the_dark(2, .5, 1, seed=1)
    scenario2 = generators.tracing_pathes_in_the_dark(2, .5, 1, seed=2)

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
        scenario1_2 = generators.tracing_pathes_in_the_dark(2, .5, 1, seed=1)
        path1_2 = storage.get_filepath(scenario1_2)
        self.assertEqual(path1, path1_2)

        # different paths for different scenarios
        path2 = storage.get_filepath(self.scenario2)
        self.assertNotEqual(path1, path2)

    def test_has_file(self):
        # has no been created yet
        self.assertFalse(storage.has_file(self.scenario1))

        # then it should have
        storage.save_result(self.scenario1, ResultType.ECBS_PATHS, 0)
        self.assertTrue(storage.has_file(self.scenario1))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
