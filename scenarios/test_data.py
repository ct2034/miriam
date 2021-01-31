from os import SEEK_CUR

import numpy as np


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
