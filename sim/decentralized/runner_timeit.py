import timeit

import numpy as np

from runner import to_agent_objects

setup = "from runner import to_agent_objects;import numpy as np"

run = "to_agent_objects(np.zeros((3, 3)), np.array(\
    [[0, 0], [2, 0]]), np.array([[2, 2], [0, 2]]))"

if __name__ == "__main__":
    t = timeit.timeit(run, setup=setup, number=10)
    print(t)
