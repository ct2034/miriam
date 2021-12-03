import os
from random import Random

from planner.dhc import generate_scenarios
from planner.dhc.DHC import test
from scenarios.generators import tracing_pathes_in_the_dark


def eval(env, starts, goals):
    fname = generate_scenarios.generate_file((env, starts, goals))
    res = test.test_model(337500, fname)
    if os.path.exists(fname):
        os.remove(fname)
    return res


if __name__ == "__main__":
    rng = Random(0)
    size = 10
    fill = .5
    n_agents = 5
    env, starts, goals = tracing_pathes_in_the_dark(
        size, fill, n_agents, rng)

    res = eval(env, starts, goals)
    print(res)
