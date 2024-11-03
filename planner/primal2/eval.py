#!/usr/bin/env python3
import os
import subprocess
from random import Random

import numpy as np

from definitions import SCENARIO_TYPE
from scenarios.evaluators import cost_ecbs
from scenarios.generators import tracing_paths_in_the_dark
from tools import hasher


def convert_scenario_to_primal(scen: SCENARIO_TYPE, path: str):
    """
    Converts a scenario to primal format.
    :param scen: scenario
    :return: path to npy file
    """
    (env, starts, goals) = scen
    workdir = os.path.dirname(__file__)

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
    np.save(path, info)


def get_fnames(workdir, scen):
    name = str(hasher(scen))
    scenario_fname = f"{workdir}/scenarios/{name}.npy"
    results_fname = f"{workdir}/testing_result/{name}_oneshotPRIMAL2.txt"
    return scenario_fname, results_fname


def eval_primal2(workdir):
    try:
        process = subprocess.Popen(
            ["./run.bash"],
            cwd=workdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
        )
        output, error = process.communicate()
        print(f"error: {error.decode('utf-8')}")
        print(f"output: {output.decode('utf-8')}")
    except Exception as e:
        print(f"error: {e}")


if __name__ == "__main__":
    workdir = os.path.dirname(__file__)
    n_eval = 10
    # generate files
    rng = Random(0)
    scens = []
    for _ in range(n_eval):
        scen = tracing_paths_in_the_dark(size=8, fill=0.5, n_agents=4, rng=rng)
        scens.append(scen)
        convert_scenario_to_primal(scen, get_fnames(workdir, scen)[0])

    # # run primal2
    # eval_primal2(workdir)

    # # read results
    # count_results = 0
    # for scen in scens:
    #     scenario_fname, results_fname = get_fnames(workdir, scen)
    #     if os.path.isfile(results_fname):
    #         # with open(results_fname, "r") as f:
    #         #     lines = f.readlines()
    #         #     for line in lines:
    #         #         print(line)
    #         count_results += 1
    #     else:
    #         print("no file")

    #     # clean up
    #     if os.path.isfile(scenario_fname):
    #         os.remove(scenario_fname)
    #     if os.path.isfile(results_fname):
    #         os.remove(results_fname)
    # print(f"{count_results}/{n_eval} scenarios evaluated")
