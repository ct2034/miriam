#!/usr/bin/env python3
import os
import subprocess
from random import Random

import numpy as np
from definitions import SCENARIO_TYPE
from scenarios.evaluators import cost_ecbs
from scenarios.generators import tracing_pathes_in_the_dark


def convert_scenario_to_primal(
        scen: SCENARIO_TYPE, name: str):
    """
    Converts a scenario to primal format.
    :param scen: scenario
    :return: path to npy file
    """
    (env, starts, goals) = scen
    cwd = os.path.dirname(__file__)

    # this should contain a
    #   -1 for every obstacle
    #   0 for every free space
    #   n+1 for start of agents n
    #   (I think)
    out_state = np.zeros(env.shape, dtype=np.int64)

    # first obstacles:
    out_state = -1 * env.astype(np.int64)

    # then starts:
    for i, start in enumerate(starts):
        out_state[tuple(start)] = i + 1

    # this should contain a
    #   0 everywhere
    #   n for goal of agents n
    #   (I think)
    out_goals = np.zeros(env.shape, dtype=np.int64)

    # then goals:
    for i, goal in enumerate(goals):
        out_goals[tuple(goal)] = i + 1

    info = np.array([out_state, out_goals])
    fname = f"{cwd}/scenarios/{name}.npy"
    np.save(fname, info)


if __name__ == '__main__':
    cwd = os.path.dirname(__file__)
    rng = Random(0)
    for _ in range(10):
        scen = tracing_pathes_in_the_dark(
            size=8,
            fill=.5,
            n_agents=4,
            rng=rng
        )
        print(f"cost_ecbs: {cost_ecbs(*scen)}")
        name = 'demo'
        try:
            convert_scenario_to_primal(scen, name)
            process = subprocess.Popen(
                ["./run.bash"],
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True)
            output, error = process.communicate()
            # print(f"error: {error}")
            print(f"output: {output}")
            # reading results:
            with open(f"{cwd}/testing_result/{name}_oneshotPRIMAL2.txt", "r") as f:
                lines = f.readlines()
                print("results:")
                for line in lines:
                    print(line)
        except Exception as e:
            print(f"error: {e}")
        finally:
            # if file exists, delete it
            if os.path.isfile(f"{cwd}/scenarios/{name}.npy"):
                os.remove(f"{cwd}/scenarios/{name}.npy")
            if os.path.isfile(f"{cwd}/testing_result/{name}_oneshotPRIMAL2.txt"):
                os.remove(f"{cwd}/testing_result/{name}_oneshotPRIMAL2.txt")
