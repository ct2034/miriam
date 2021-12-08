#!/usr/bin/env python3
import random

import numpy as np
from matplotlib import pyplot as plt
from planner.mapf_with_rl.mapf_with_rl import Scenario
from scenarios.visualization import (plot_state, plot_env_with_arrows,
                                     plot_with_paths)

if __name__ == "__main__":
    env = np.array([
        [0, 1, 1, 1],
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0]
    ])
    starts = np.array([
        [0, 0],
        [3, 3],
        [2, 3]
    ])
    goals = np.array([
        [3, 2],
        [0, 0],
        [3, 0]
    ])
    s = Scenario(
        (env, starts, goals),
        ignore_finished_agents=True,
        hop_dist=3,
        rng=random.Random(0)
    )
    assert s.useful
    plot_env_with_arrows(env, starts, goals)

    paths = list(map(lambda a: np.array(a.path), s.agents))
    plot_with_paths(env, paths)

    state = s.start()
    plot_state(state)
    plt.show()
