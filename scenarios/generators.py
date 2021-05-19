#!/usr/bin/env python3
import logging
import os
import random
from itertools import product, repeat
from typing import *

import numpy as np
import sim.decentralized.agent
import sim.decentralized.runner
import tools
from definitions import FREE, OBSTACLE
from sim.decentralized.policy import PolicyType

logging.getLogger('sim.decentralized.agent').setLevel(logging.ERROR)


def make_starts_goals_on_env(env: np.ndarray, n_agents: int,
                             seed: float = random.random()):
    random.seed(seed)
    agents = sim.decentralized.runner.initialize_agents(
        env, n_agents, PolicyType.RANDOM,
        tight_placement=True, seed=seed)
    assert agents is not None
    starts = np.array([a.pos for a in agents])
    assert starts.shape == (n_agents, 2)
    goals = np.array([a.goal for a in agents])
    assert goals.shape == (n_agents, 2)
    return starts, goals


# random ######################################################################

def like_sim_decentralized(size: int, fill: float,
                           n_agents: int, seed: float):
    """Randomly filling spaces in gridmap based on `fill`."""
    random.seed(seed)
    env = sim.decentralized.runner.initialize_environment(
        size, fill, seed=seed)
    starts, goals = make_starts_goals_on_env(env, n_agents, seed=seed)
    return env, starts, goals


# this was previously in planner.policylearn.generate_data
def generate_random_gridmap(width: int, height: int, fill: float):
    """Making a random gridmap of size (`width`x`height`). It will be filled
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


def like_policylearn_gen(size: int, fill: float,
                         n_agents: int, seed: float):
    random.seed(seed)
    env = generate_random_gridmap(
        size, size, fill)
    starts, goals = make_starts_goals_on_env(env, n_agents)
    return env, starts, goals

# tracing pathes in the dark ##################################################


def get_random_next_to_free_pose_or_any_if_full(env):
    """If there are free spaces in the map, return a random free pose.
    From that we step in a random direction.
    If map is fully black, return any pose."""
    size = env.shape[0]
    if np.sum(env == FREE) == 0:  # all obstacle
        samples = np.where(env == OBSTACLE)
    else:
        samples = np.where(env == FREE)
    n_samples = len(samples[1])
    r = random.randrange(n_samples)
    basic_pos = np.array(samples)[:, r]
    r_step = random.choice([[0, 1], [0, -1], [1, 0], [-1, 0]])
    step_pos = basic_pos + r_step
    if (step_pos[0] < 0 or step_pos[1] < 0 or
            step_pos[0] >= size or step_pos[1] >= size):
        return basic_pos
    else:
        return step_pos


def tracing_pathes_in_the_dark(size: int, fill: float,
                               n_agents: int, seed: float):
    """Starting with a black map, clearing straight lines through it, making
    sure map is fully connected."""
    if fill == 0:
        env = np.zeros((size, size), dtype=np.int8)
    else:
        random.seed(seed)
        np.random.seed(int(seed*1000))
        env = np.ones((size, size), dtype=np.int8)
        to_clear_start = int((1. - fill) * size * size)
        to_clear = to_clear_start
        while to_clear > 0:
            env = np.rot90(env)
            start = get_random_next_to_free_pose_or_any_if_full(env)
            dist = min(random.randrange(size), to_clear)
            env[start[0]:dist, start[1]] = FREE
            to_clear = to_clear_start - np.sum(env == FREE)
    starts, goals = make_starts_goals_on_env(env, n_agents, seed)
    return env, starts, goals


# movingai ####################################################################

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
    scene_files: List[str] = []
    scene_nr_str = "{}-{:d}.".format(scene_str, scene_nr+1)
    for subdir, dirs, files in os.walk(MOVINGAI_PATH):
        for filename in files:
            filepath = subdir + os.sep + filename
            if filepath.endswith(map_str + ".map"):
                mapfile = filepath
            if (filepath.endswith(".scen") and
                map_str in filepath and
                    scene_nr_str in filepath):
                scene_files.append(filepath)
    print("\n".join(scene_files))

    # reading mapfile
    print(mapfile)
    grid = movingai_read_mapfile(mapfile)

    # reading scene_file
    assert len(scene_files) == 1
    print(scene_files[0])
    with open(scene_files[0], 'r') as f:
        scene_file_content = f.read().split("\n")
    assert scene_file_content[0] == "version 1"
    max_n_jobs = len(scene_file_content) - 2
    assert max_n_jobs >= n_agents
    jobs = np.zeros((n_agents, 4), dtype=np.int32)

    starts = []
    goals = []
    LINES_OFFSET = 1
    for i_l in range(n_agents):
        line = scene_file_content[i_l + LINES_OFFSET]
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


# building walls ##############################################################

def no_diagonals(area):
    """Make sure this area will produce no diagonal walls,"""
    return area[0, 1] or area[2, 1] or area[1, 0] or area[1, 2]


def can_area_be_set(area):
    """Can the pixel in the middle of this 3x3 are be set as an obstacle
    without making disconnected areas in the map."""
    assert area.shape[0] == 3
    assert area.shape[1] == 3
    assert area[1, 1] == FREE
    sequence = area[
        [0, 0, 0, 1, 2, 2, 2, 1],
        [0, 1, 2, 2, 2, 1, 0, 0]
    ]
    diffs = np.where(sequence != np.roll(sequence, 1))
    return (len(diffs[0]) <= 2 and
            no_diagonals(area))


def can_be_set(env, pos):
    """Can the pixel at pos in env be set without making disconnected areas,"""
    assert len(pos) == 2
    assert pos[0] < env.shape[0]
    assert pos[1] < env.shape[1]
    area = np.full((3, 3), OBSTACLE)
    for x, y in product(range(3), repeat=2):
        padded_env = np.array(pos) + [x, y] + [-1, -1]
        if (
                padded_env[0] < 0 or padded_env[0] >= env.shape[0] or
                padded_env[1] < 0 or padded_env[1] >= env.shape[1]
        ):
            area[x, y] = OBSTACLE
        else:
            area[x, y] = env[tuple(padded_env)]
    return can_area_be_set(area)


def building_walls(size: int, fill: float,
                   n_agents: int, seed: float):
    """Starting with an empty map, creating obstacles inline of previous
    obstacles, ensuring full connectedness."""
    if fill == 0:
        env = np.zeros((size, size), dtype=np.int8)
    else:
        random.seed(seed)
        np.random.seed(int(seed*1000))
        env = np.full((size, size), FREE, dtype=np.int8)
        to_fill = int(fill * size ** 2)
        while np.sum(env == OBSTACLE) < to_fill:
            free = np.where(env == FREE)
            can_fill = False
            pos = [0, 0]
            while not can_fill:
                i_f = random.randrange(len(free[0]))
                pos[0] = free[0][i_f]
                pos[1] = free[1][i_f]
                can_fill = can_be_set(env, pos)
            env[tuple(pos)] = OBSTACLE
    starts, goals = make_starts_goals_on_env(env, n_agents, seed)
    return env, starts, goals
