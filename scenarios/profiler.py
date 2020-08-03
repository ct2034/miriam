import cProfile
import numpy as np
from scenarios.generators import like_policylearn_gen

import cachier


def some_gridmaps():
    for fill in np.arange(0, .8, 100):
        like_policylearn_gen(10, fill, 16, 99)


cProfile.run("some_gridmaps()")

# from scenarios.eval_wellformed_ecbs import main

# cProfile.run("main()")
