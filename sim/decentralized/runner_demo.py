#!/usr/bin/env python3
import logging
import random

import numpy as np
from matplotlib import pyplot as plt

from definitions import INVALID
from scenarios.evaluators import (
    cost_ecbs,
    cost_independent,
    cost_sim_decentralized_learned,
)
from scenarios.generators import (
    arena_with_crossing,
    corridor_with_passing,
    random_fill,
    tracing_pathes_in_the_dark,
)
from scenarios.solvers import ecbs
from scenarios.visualization import plot_env_with_arrows, plot_with_paths
from sim.decentralized.iterators import IteratorType
from sim.decentralized.policy import PolicyType
from sim.decentralized.runner import run_a_scenario, to_agent_objects
from tools import ProgressBar


def arena():
    # scenario arena
    rng = random.Random(0)
    (env, starts, goals) = arena_with_crossing(10, 0, 4, rng)
    print(starts)
    print(goals)
    plot_env_with_arrows(env, starts, goals)
    plt.show()
    results = []
    n_runs = 100
    pb = ProgressBar("demo", n_runs, 10)
    for _ in range(n_runs):
        (env, starts, goals) = arena_with_crossing(10, 0, 6, rng)
        agents = to_agent_objects(env, starts, goals, PolicyType.RANDOM, rng)
        res = run_a_scenario(
            env, agents, False, IteratorType.LOOKAHEAD1, ignore_finished_agents=False
        )
        results.append(res)
        pb.progress()
    pb.end()
    print(f"average_success_rate={np.mean([r[4] for r in results]):.3f}")


def corridor():
    # scenario with corridor and one place to pas
    logging.getLogger("sim.decentralized.runner").setLevel(logging.DEBUG)
    rng = random.Random(0)
    (env, starts, goals) = corridor_with_passing(10, 0, 2, rng)
    plot_env_with_arrows(env, starts, goals)
    results = []
    n_runs = 100
    pb = ProgressBar("demo", n_runs, 10)
    for i_r in range(n_runs):
        print(f"run {i_r}")
        rng = random.Random(i_r)
        (env, starts, goals) = corridor_with_passing(10, 0, 2, rng)
        agents = to_agent_objects(
            env, starts, goals, PolicyType.RANDOM, radius=0.3, rng=rng
        )
        if agents is not None:
            res = run_a_scenario(
                env,
                tuple(agents),
                False,
                IteratorType.LOOKAHEAD2,
                ignore_finished_agents=False,
            )
        else:
            res = (0,) * 5
        results.append(res)
        pb.progress()
    pb.end()
    print(f"average_success_rate={np.mean([r[4] for r in results]):.3f}")


def a_scenario():
    # a scenario
    env = np.array([[0, 0, 0, 1], [0, 0, 0, 0], [1, 1, 0, 1], [0, 0, 0, 0]])
    starts = np.array([[0, 2], [1, 1], [3, 3]])
    goals = np.array([[0, 1], [3, 1], [1, 1]])
    agents = to_agent_objects(env, starts, goals, rng=random.Random(0))
    # plot_with_arrows(env, starts, goals)
    # plt.show()
    res = run_a_scenario(
        env,
        agents,
        False,
        IteratorType.LOOKAHEAD1,
        ignore_finished_agents=False,
        print_progress=True,
    )
    print(res)


def big_environments():
    # checking big environments
    env, starts, goals = random_fill(40, 0.2, 50, random.Random(0))
    agents = to_agent_objects(env, starts, goals)
    plot_env_with_arrows(env, starts, goals)
    plt.show()
    # paths_ecbs = ecbs(env, starts, goals, return_paths=True)
    # plot_with_paths(env, paths_ecbs)
    res = run_a_scenario(
        env, agents, False, IteratorType.LOOKAHEAD1, ignore_finished_agents=False
    )
    print(res)


def blocking_3():
    env = np.zeros((3, 3))
    starts = np.array([[0, 0], [2, 0]])
    goals = np.array([[2, 2], [0, 2]])
    agents = to_agent_objects(env, starts, goals)
    paths_ecbs = ecbs(env, starts, goals, return_paths=True)
    plot_with_paths(env, paths_ecbs)
    assert agents is not None
    res = run_a_scenario(env, tuple(agents), True, IteratorType.LOOKAHEAD3)
    print(res)


def find_and_run_interesting_scenario():
    logging.getLogger("sim.decentralized.agent").setLevel(logging.ERROR)
    logging.getLogger("__main__").setLevel(logging.ERROR)
    logging.getLogger("root").setLevel(logging.ERROR)

    size = 8
    n_agents = 8
    policy = PolicyType.LEARNED
    it = IteratorType.LOOKAHEAD3
    interesting = False
    seed = 0

    while not interesting:
        env, starts, goals = tracing_pathes_in_the_dark(
            size, 0.5, n_agents, random.Random(seed)
        )
        agents = to_agent_objects(env, starts, goals, policy)
        c_ecbs = cost_ecbs(env, starts, goals)
        c_indep = cost_independent(env, starts, goals)
        c_decen = cost_sim_decentralized_learned(env, starts, goals, skip_cache=True)
        print(f"seed: {seed}")
        print(f"c_ecbs: {c_ecbs}, c_indep: {c_indep}, c_decen: {c_decen}")
        if c_ecbs != INVALID and c_indep != INVALID and c_decen != INVALID:
            interesting = c_decen > c_ecbs
        seed += 1

    res_ecbs = ecbs(env, starts, goals, return_paths=False)
    print(res_ecbs["blocks"])
    paths_ecbs = ecbs(env, starts, goals, return_paths=True)
    plot_with_paths(env, paths_ecbs)
    plt.show()

    logging.getLogger("sim.decentralized.policy").setLevel(logging.DEBUG)
    res_decen = run_a_scenario(env, agents, plot=True, iterator=it)
    print(res_decen)


if __name__ == "__main__":  # pragma: no cover
    # arena()
    corridor()
    # a_scenario()
    # big_environments()
    # blocking_3()
    # find_and_run_interesting_scenario()
