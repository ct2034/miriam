import numpy as np

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

# starts and goals with a collision in middle of env
starts_collision = np.array([
    [0, 0],
    [0, 2]
])
goals_collision = np.array([
    [2, 0],
    [2, 2]
])

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
