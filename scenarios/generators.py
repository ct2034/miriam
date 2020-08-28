#!/usr/bin/env python3
import logging
import random
from itertools import product
from typing import *

from cachier import cachier
import numpy as np
from matplotlib import pyplot as plt

import sim.decentralized.agent
import sim.decentralized.runner
import tools

logging.getLogger('sim.decentralized.agent').setLevel(logging.ERROR)

FREE = 0
OBSTACLE = 1


def make_starts_goals_on_env(env: np.ndarray, n_agents: int):
    agents = sim.decentralized.runner.initialize_agents(
        env, n_agents, sim.decentralized.agent.Policy.RANDOM)
    starts = np.array([a.pos for a in agents])
    assert starts.shape == (n_agents, 2)
    goals = np.array([a.goal for a in agents])
    assert goals.shape == (n_agents, 2)
    return starts, goals


@cachier(hash_params=tools.hasher)
def like_sim_decentralized(size: int, fill: float,
                           n_agents: int, seed: Any):
    random.seed(seed)
    env = sim.decentralized.runner.initialize_environment(size, fill)
    starts, goals = make_starts_goals_on_env(env, n_agents)
    return env, starts, goals


# this was previously in planner.policylearn.generate_data
def generate_random_gridmap(width: int, height: int, fill: float):
    """making a random gridmap of size (`width`x`height`). It will be filled
    with stripes until `fill` is exceeded and then single cells are freed until
    `fill` is exactly reached."""
    gridmap = np.zeros((width, height), dtype=np.int8)
    while np.count_nonzero(gridmap) < fill * width * height:
        direction = random.randint(0, 1)
        start = (
            random.randint(0, width-1),
            random.randint(0, height-1)
        )
        if direction:  # x
            gridmap[start[0]:random.randint(0, width-1), start[1]] = OBSTACLE
        else:  # y
            gridmap[start[0], start[1]:random.randint(0, height-1)] = OBSTACLE
    while np.count_nonzero(gridmap) > fill * width * height:
        make_free = (
            random.randint(0, width-1),
            random.randint(0, height-1)
        )
        gridmap[make_free] = FREE
    return gridmap


@cachier(hash_params=tools.hasher)
def like_policylearn_gen(size: int, fill: float,
                         n_agents: int, seed: Any):
    random.seed(seed)
    env = generate_random_gridmap(
        size, size, fill)
    starts, goals = make_starts_goals_on_env(env, n_agents)
    return env, starts, goals


def get_random_next_to_free_pose_or_any_if_full(env):
    """If there are free spaces in the map, return a random free pose.
    From that we step in a random direction.
    If map is fully black, return any pose."""
    size = env.shape[0]
    if np.sum(env == FREE) == 0:  # all obstacle
        samples = np.where(env == OBSTACLE)
    else:
        samples = np.where(env == FREE)
    n_smpl = len(samples[1])
    r = random.randrange(n_smpl)
    basic_pos = np.array(samples)[:, r]
    r_step = random.choice([[0, 1], [0, -1], [1, 0], [-1, 0]])
    step_pos = basic_pos + r_step
    if (step_pos[0] < 0 or step_pos[1] < 0 or
            step_pos[0] >= size or step_pos[1] >= size):
        return basic_pos
    else:
        return step_pos


def tracing_pathes_in_the_dark(size: int, fill: float,
                               n_agents: int, seed: Any):
    if fill == 0:
        env = np.zeros((size, size), dtype=np.int8)
    else:
        random.seed(seed)
        env = np.ones((size, size), dtype=np.int8)
        to_clear_start = int((1. - fill) * size * size)
        to_clear = to_clear_start
        while to_clear > 0:
            direction = random.randint(0, 1)
            start = get_random_next_to_free_pose_or_any_if_full(env)
            dist = min(random.randrange(size), to_clear)
            if direction:  # x
                env[start[0]:dist, start[1]] = FREE
            else:  # y
                env[start[0], start[1]:dist] = FREE
            to_clear = to_clear_start - np.sum(env == FREE)
    starts, goals = make_starts_goals_on_env(env, n_agents)
    return env, starts, goals
