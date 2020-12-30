#!/usr/bin/env python3
import logging
import random
from itertools import product
from typing import *
import os

from cachier import cachier
import numpy as np
from matplotlib import pyplot as plt

import sim.decentralized.agent
import sim.decentralized.runner
import tools
from definitions import FREE, OBSTACLE

logging.getLogger('sim.decentralized.agent').setLevel(logging.ERROR)


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


def movingai_read_mapfile(mapfile: str):
    FREE_CHAR = "."
    OBSTACLE_CHAR = "@"
    def decode(c): return FREE if c == FREE_CHAR else OBSTACLE
    with open(mapfile, 'r') as f:
        mapfile_content = f.read().split("\n")
    assert mapfile_content[0] == "type octile"
    assert mapfile_content[1].startswith("height")
    height = int(mapfile_content[1].split(" ")[1])
    assert mapfile_content[2].startswith("width")
    width = int(mapfile_content[2].split(" ")[1])
    assert mapfile_content[3] == "map"
    LINES_OFFSET = 4

    grid = np.zeros((width, height), dtype=np.int8)
    for i_l in range(height):
        grid[:, i_l] = list(map(decode, mapfile_content[i_l + LINES_OFFSET]))
    return grid


def movingai(map_str: str, scene_str: str, scene_nr: int, n_agents: int):
    MOVINGAI_PATH = 'scenarios/movingai'

    assert (scene_str == "even" or scene_str == "random"
            ), "scene_str may be either >even< or >random<"
    scenfiles: List[str] = []
    scen_nr_str = "{}-{:d}.".format(scene_str, scene_nr+1)
    for subdir, dirs, files in os.walk(MOVINGAI_PATH):
        for filename in files:
            filepath = subdir + os.sep + filename
            if filepath.endswith(map_str + ".map"):
                mapfile = filepath
            if (filepath.endswith(".scen") and
                map_str in filepath and
                    scen_nr_str in filepath):
                scenfiles.append(filepath)
    print("\n".join(scenfiles))

    # reading mapfile
    print(mapfile)
    grid = movingai_read_mapfile(mapfile)

    # reading scenfile
    assert len(scenfiles) == 1
    print(scenfiles[0])
    with open(scenfiles[0], 'r') as f:
        scenfile_content = f.read().split("\n")
    assert scenfile_content[0] == "version 1"
    max_n_jobs = len(scenfile_content) - 2
    assert max_n_jobs >= n_agents
    jobs = np.zeros((n_agents, 4), dtype=np.int32)

    starts = []
    goals = []
    LINES_OFFSET = 1
    for i_l in range(n_agents):
        line = scenfile_content[i_l + LINES_OFFSET]
        elem = line.split("\t")
        assert map_str in elem[1]
        # start
        start = [int(elem[4]), int(elem[5])]
        assert grid[tuple(start)] == FREE
        starts.append(start)
        # goal
        goal = [int(elem[6]), int(elem[7])]
        assert grid[tuple(goal)] == FREE
        goals.append(goal)

    return grid, starts, goals
