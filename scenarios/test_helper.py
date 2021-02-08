import os
import shutil
import uuid

import numpy as np

ENVVAR_STORAGE_PATH_STR = 'SCENARIO_STORAGE_PATH'
STORAGE_PATH_TESTING = "/tmp/testing/"


def make_cache_folder_and_set_envvar(set_envvar=True):
    if not os.path.exists(STORAGE_PATH_TESTING):
        os.mkdir(STORAGE_PATH_TESTING)
    assert not os.listdir(
        STORAGE_PATH_TESTING), "testing cache folder is not empty"
    data_path = STORAGE_PATH_TESTING + str(uuid.uuid1())
    assert not os.path.exists(data_path)
    os.mkdir(data_path)
    if set_envvar:
        os.environ[ENVVAR_STORAGE_PATH_STR] = data_path
    print("folder for testing created under " + data_path)
    return data_path


def remove_cache_folder_and_unset_envvar(unset_envvar=True):
    assert (ENVVAR_STORAGE_PATH_STR in os.environ
            ), "environment variable must be set"
    data_path = os.environ[ENVVAR_STORAGE_PATH_STR]
    shutil.rmtree(data_path)
    if unset_envvar:
        del os.environ[ENVVAR_STORAGE_PATH_STR]
    print("folder for testing deleted under " + data_path)


def assert_path_equality(self, should_be, test):
    self.assertTrue(isinstance(should_be, list))
    self.assertTrue(isinstance(test, list))
    self.assertEqual(len(should_be), len(test))
    n_agents = len(should_be)
    for i_a in range(n_agents):
        self.assertTrue(np.all(should_be[i_a] == test[i_a]))


env = np.array([
    [0, 0, 0],
    [1, 0, 1],
    [0, 0, 0]
])

# starts and goals with no collision on env
starts_no_collision = np.array([
    [0, 0],
    [2, 0]
])
goals_no_collision = np.array([
    [0, 2],
    [2, 2]
])
paths_no_collision = [
    np.array([
        [0, 0, 0],
        [0, 1, 1],
        [0, 2, 2]
    ]),
    np.array([
        [2, 0, 0],
        [2, 1, 1],
        [2, 2, 2]
    ])


]

# starts and goals with a collision in middle of env
starts_collision = np.array([
    [0, 0],
    [0, 2]
])
goals_collision = np.array([
    [2, 0],
    [2, 2]
])
paths_collision_indep = [
    np.array([
        [0, 0, 0],
        [0, 1, 1],
        [1, 1, 2],
        [2, 1, 3],
        [2, 0, 4]
    ]),
    np.array([
        [0, 2, 0],
        [0, 1, 1],
        [1, 1, 2],
        [2, 1, 3],
        [2, 2, 4]
    ])
]

# narrow hallway with two agents that can not pass each other
env_deadlock = np.array([
    [1, 1, 1],
    [0, 0, 0],
    [1, 1, 1]
])
starts_deadlock = np.array([
    [1, 0],
    [1, 2]
])
goals_deadlock = np.array([
    [1, 2],
    [1, 0]
])

# complicated example
env_complicated = np.array([
    [0, 0, 0, 0],
    [1, 1, 0, 0],
    [0, 1, 0, 1],
    [0, 0, 0, 0]
])
starts_complicated = np.array([
    [0, 0],
    [0, 3],
    [3, 0]
])
goals_complicated = np.array([
    [3, 0],
    [3, 3],
    [0, 0]
])
