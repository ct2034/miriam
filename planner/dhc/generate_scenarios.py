import pickle
from random import Random

from definitions import SCENARIO_TYPE
from scenarios.generators import tracing_pathes_in_the_dark
from tools import hasher
from tqdm import tqdm


def generate_file(scen: SCENARIO_TYPE):
    """
    Generate a file for Scenario scen.
    :param scen: The type of scenario to generate.
    """
    hashnr = abs(hash(hasher(scen)))
    fname = f"planner/dhc/DHC/test_set/{hashnr}.pth"
    tests = [scen]
    with open(fname, 'wb') as f:
        pickle.dump(tests, f)
    return fname


if __name__ == "__main__":
    n_scenarios = 10
    tests = []
    rng = Random(0)

    size = 10
    fill = .5
    n_agents = 5

    for _ in tqdm(range(n_scenarios)):
        env, starts, goals = tracing_pathes_in_the_dark(
            size, fill, n_agents, rng)
        tests.append((env, starts, goals))

    path = f'./DHC/test_set/tracing_pathes_in_the_dark_{size}_{fill}_{n_agents}.pth'
    with open(path, 'wb') as f:
        pickle.dump(tests, f)
