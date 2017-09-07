import os
import numpy as np


def read_file(fname, grid):
    height = grid.shape[1]
    path = os.getenv("COBRA_PATH")
    if path:
        fname = path + "/" + fname
    paths = []
    agent_path = []
    t = 0
    with open(fname, 'r') as f:
        for line in f:
            nums = line.strip().split("\t")
            if len(nums) == 1:  # a new agent
                n = int(nums[0])
                if agent_path:
                    paths.append((agent_path,))
                    agent_path = []
                    t = 0
            else:  # a new point
                agent_path.append(
                    (int(nums[0]),
                     height - int(nums[1]),
                     t)
                )
                t += 1
        paths.append((agent_path,))
    return paths
