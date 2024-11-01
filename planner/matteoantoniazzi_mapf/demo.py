import numpy as np
from planner.matteoantoniazzi_mapf.plan import icts
from tools import get_map_str

if __name__ == "__main__":
    env = np.zeros([4, 6])
    env[1:4, 1] = 1
    env[1:3, 3] = 1
    env[1:4, 5] = 1
    print(get_map_str(env))

    starts = np.array([[0, 0], [0, 3], [4, 3], [5, 0], [4, 2]])

    goals = np.array([[3, 3], [5, 0], [3, 0], [0, 0], [0, 2]])

    info = icts(env, starts, goals)
    print(info)
