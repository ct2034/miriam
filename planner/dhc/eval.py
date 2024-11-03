import os
from random import Random

from definitions import INVALID
from planner.dhc import generate_scenarios
from planner.dhc.DHC import test
from scenarios.generators import tracing_paths_in_the_dark


def eval(env, starts, goals):
    fname = generate_scenarios.generate_file((env, starts, goals))
    res = test.test_model(337500, fname)
    if os.path.exists(fname):
        os.remove(fname)
    if res is None:
        return INVALID
    elif res[0] is None:
        return INVALID
    elif res[0][0] == False:
        return INVALID
    else:
        return res[0]  # only one case was tested


if __name__ == "__main__":
    rng = Random(0)
    size = 10
    fill = 0.5
    n_agents = 5
    env, starts, goals = tracing_paths_in_the_dark(size, fill, n_agents, rng)
    print(f"{starts=}")
    print(f"{goals=}")

    res = eval(env, starts, goals)
    assert isinstance(res, tuple)
    success, steps, paths = res
    print(f"{success=}\n{steps=}\n{paths=}")
