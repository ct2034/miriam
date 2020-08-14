#!/usr/bin/env python3

import argparse
import logging
import pickle
import random
import sys

from cachier import cachier
import matplotlib.pyplot as plt
import numpy as np

import tools
from planner.policylearn.libMultiRobotPlanning.plan_ecbs import (
    BLOCKS_STR, plan_in_gridmap)
from sim.decentralized.runner import initialize_environment
from scenarios.generators import tracing_pathes_in_the_dark

FREE = 0
OBSTACLE = 1
VERTEX_CONSTRAINTS_STR = 'vertexConstraints'
EDGE_CONSTRAINTS_STR = 'edgeConstraints'
SCHEDULE_STR = 'schedule'
INDEP_AGENT_PATHS_STR = 'indepAgentPaths'
COLLISIONS_STR = 'collisions'
GRIDMAP_STR = 'gridmap'
OWN_STR = 'own'
OTHERS_STR = 'others'
TRANSFER_LSTM_STR = 'transfer_lstm'
TRANSFER_CLASSIFICATION_STR = 'transfer_classification'
GENERATE_SIM_STR = 'generate_simulation'

LSTM_FOV_RADIUS = 2  # self plus x in all 4 directions
CLASSIFICATION_POS_TIMESTEPS = 3
CLASSIFICATION_FOV_RADIUS = 6  # self plus x in all 4 directions
DTYPE_SAMPLES = np.int8


def show_map(x):
    """displays the map using matplotlib. You may want to call `plt.show()`
    yourself."""
    plt.imshow(
        np.swapaxes(x, 0, 1),
        aspect='equal',
        cmap=plt.get_cmap("binary"),
        origin='lower')


def get_random_free_pos(gridmap):
    """return a random pose from that map, that is free."""
    free = np.where(gridmap == FREE)
    choice = random.randint(0, len(free[0])-1)
    pos = tuple(np.array(free)[:, choice])
    return pos


def get_vertex_block_coords(blocks):  # TODO: edge constraints
    """from the blocks section of data, get all vertex blocks."""
    coords = []
    for agent in blocks.keys():
        coords_pa = []
        if blocks[agent]:
            blocks_pa = blocks[agent]
            if VERTEX_CONSTRAINTS_STR in blocks_pa.keys():
                for vc in blocks_pa[VERTEX_CONSTRAINTS_STR]:
                    coords_pa.append([vc['v.x'], vc['v.y']])
        coords.append(np.array(coords_pa))
    return coords


def count_blocks(blocks):
    """counts if edge and vertex blocks"""
    n_vc = 0
    n_ec = 0
    for agent in blocks.keys():
        if blocks[agent]:
            blocks_pa = blocks[agent]
            if VERTEX_CONSTRAINTS_STR in blocks_pa.keys():
                for _ in blocks_pa[VERTEX_CONSTRAINTS_STR]:
                    n_vc += 1
            if EDGE_CONSTRAINTS_STR in blocks_pa.keys():
                for _ in blocks_pa[EDGE_CONSTRAINTS_STR]:
                    n_ec += 1
    return n_vc, n_ec


def has_exactly_one_vertex_block(blocks):
    """counts if these blocks have exactly one vertex block
    (and no edge block)."""
    n_vc, n_ec = count_blocks(blocks)
    return n_ec == 0 and n_vc == 1


def has_one_or_more_vertex_blocks(blocks):
    """counts if these blocks have one or more vertex blocks."""
    n_vc, n_ec = count_blocks(blocks)
    return n_vc >= 1


def get_agent_paths_from_data(data, timed=False):
    """get paths for all agents from the data dict. Can be chosen to be
    with time [x, y, t] (timed=True) or only the positions."""
    agent_paths = []
    if not data:
        return []
    elif SCHEDULE_STR in data.keys():
        schedule = data[SCHEDULE_STR]
        for agent_str in schedule.keys():
            path_pa = []
            schedule_pa = schedule[agent_str]
            for pose in schedule_pa:
                if timed:
                    path_pa.append([
                        pose['x'],
                        pose['y'],
                        pose['t']
                    ])
                else:
                    path_pa.append([
                        pose['x'],
                        pose['y']
                    ])
            agent_paths.append(np.array(path_pa, dtype=DTYPE_SAMPLES))
    return agent_paths


@cachier(hash_params=tools.hasher)
def will_they_collide(gridmap, starts, goals):
    """checks if for a given set of starts and goals the agents travelling
    between may collide on the given gridmap."""
    collisions = {}
    seen = set()
    been_at = {}
    agent_paths = []
    for i_a, _ in enumerate(starts):
        data = plan_in_gridmap(gridmap, [starts[i_a], ], [
                               goals[i_a], ], timeout=2)
        if data is None:
            logging.warn("no single agent plan in gridmap")
            plt.show()
            return {}, []
        single_agent_paths = get_agent_paths_from_data(data, timed=True)
        if not single_agent_paths:
            logging.warn("no single agent path from ecbs data")
            return {}, []
        agent_paths.append(single_agent_paths[0])
        for pos in single_agent_paths[0]:
            pos = tuple(pos)
            if pos in seen:  # TODO: edge collisions
                collisions[pos] = (been_at[pos], i_a)
            seen.add(pos)
            been_at[pos] = i_a
    return collisions, agent_paths


@cachier(hash_params=tools.hasher)
def add_padding_to_gridmap(gridmap, radius):
    """add a border of blocks around the map of given radius.
    (The new size will be old size + 2 * radius in both directions)"""
    size = gridmap.shape
    padded_gridmap = np.ones([
        size[0] + 2 * radius,
        size[1] + 2 * radius],
        dtype=DTYPE_SAMPLES)
    padded_gridmap[
        radius:size[0]+radius,
        radius:size[1]+radius] = gridmap
    return padded_gridmap


def training_samples_from_data(data, mode):
    """extract training samples from the data simulation data dict."""
    training_samples = []
    n_agents = len(data[INDEP_AGENT_PATHS_STR])
    # assert len(data[COLLISIONS_STR]
    #            ) == 1, "assuming we only handle one conflict"
    for col_vertex, col_agents in data[COLLISIONS_STR].items():
        t = col_vertex[2]
        blocked_agent = -1
        unblocked_agent = -1
        for i_a in col_agents:
            if data[BLOCKS_STR]["agent"+str(i_a)] is dict:
                blocked_agent = i_a
            else:
                unblocked_agent = i_a
        if mode == TRANSFER_LSTM_STR:
            data_pa = []
            training_samples.extend(lstm_samples(
                n_agents, data, t, data_pa, col_agents, unblocked_agent))
        elif mode == TRANSFER_CLASSIFICATION_STR:
            training_samples.extend(classification_samples(
                n_agents, data, t, col_agents, unblocked_agent))
    return training_samples


def lstm_samples(n_agents, data, t, data_pa, col_agents, unblocked_agent):
    """specifically construct training data for the lstm model."""
    training_samples = []
    padded_gridmap = add_padding_to_gridmap(data[GRIDMAP_STR], LSTM_FOV_RADIUS)
    for i_a in range(n_agents):
        # training features
        path = get_path(data[INDEP_AGENT_PATHS_STR][i_a], t)
        fovs = make_obstacle_fovs(padded_gridmap,
                                  path,
                                  t,
                                  LSTM_FOV_RADIUS)
        deltas = make_target_deltas(path,
                                    t)
        data_this_agent = []
        for i_t in range(t+1):
            data_this_agent.append(
                np.append(
                    np.concatenate(fovs[i_t]),
                    [
                        deltas[i_t],
                        path[i_t][:2]
                    ]
                )
            )
        data_pa.append(data_this_agent)
    for i_a in col_agents:
        x = {
            OWN_STR: data_pa[i_a],
            OTHERS_STR: [data_pa[i_ao]
                         for i_ao in range(n_agents) if i_a != i_ao]
        }
        training_samples.append((
            x,
            1 if i_a == unblocked_agent else 0))
    return training_samples


def classification_samples(n_agents, data, t, col_agents,
                           unblocked_agent):
    """specifically construct training data for the classification model."""
    training_samples = []
    paths_until_col = []
    paths_full = []
    padded_gridmap = add_padding_to_gridmap(data[GRIDMAP_STR],
                                            CLASSIFICATION_FOV_RADIUS)
    for i_a in range(n_agents):
        path_until_col = get_path(data[INDEP_AGENT_PATHS_STR][i_a], t)
        if len(path_until_col) < CLASSIFICATION_POS_TIMESTEPS:
            padded_path = np.zeros([CLASSIFICATION_POS_TIMESTEPS, 2])
            for i in range(
                0, CLASSIFICATION_POS_TIMESTEPS - len(path_until_col)
            ):
                padded_path[i] = path_until_col[0]
            padded_path[i+1:] = path_until_col
            path_until_col = padded_path
        paths_until_col.append(path_until_col[-CLASSIFICATION_POS_TIMESTEPS:])
        # full path:
        path_full = get_path(data[INDEP_AGENT_PATHS_STR][i_a], -1)
        paths_full.append(path_full)
    t = CLASSIFICATION_POS_TIMESTEPS-1
    for i_a in col_agents:
        obstacle_fovs = make_obstacle_fovs(
            padded_gridmap, paths_until_col[i_a], t, CLASSIFICATION_FOV_RADIUS)
        pos_other_agent_fovs = make_other_agent_fovs(
            paths_until_col, i_a, CLASSIFICATION_FOV_RADIUS)
        path_fovs, paths_other_agents_fovs = make_path_fovs(
            paths_full, paths_until_col, i_a, t, CLASSIFICATION_FOV_RADIUS)
        x = np.stack([obstacle_fovs, pos_other_agent_fovs,
                      path_fovs, paths_other_agents_fovs], axis=3)
        training_samples.append((
            x,
            1 if i_a == unblocked_agent else 0))
    return training_samples


def make_obstacle_fovs(padded_gridmap, path, t, radius):
    """create for all agents a set of FOVS of radius containing positions of
    obstacles in gridmap."""
    obstacle_fovs = []
    for i_t in range(t+1):
        pos = path[i_t]
        obstacle_fovs.append(
            padded_gridmap[
                int(pos[0]):int(pos[0]) + 1 + 2 * radius,
                int(pos[1]):int(pos[1]) + 1 + 2 * radius
            ]
        )
    obstacle_fovs_np = np.stack(obstacle_fovs, axis=2)
    return obstacle_fovs_np


def make_other_agent_fovs(paths, agent, radius):
    """create for the agent a set of FOVS of radius containing positions of
    other agents."""
    t = paths[0].shape[0]
    other_agent_fovs = init_empty_fov(radius, t)
    for i_t in range(t):
        pos = paths[agent][i_t]
        for i_a in [i for i in range(len(paths)) if i != agent]:
            d = paths[i_a][i_t] - pos
            if (abs(d[0]) <= CLASSIFICATION_FOV_RADIUS and
                    abs(d[1]) <= CLASSIFICATION_FOV_RADIUS):
                other_agent_fovs[
                    int(d[0]) + CLASSIFICATION_FOV_RADIUS,
                    int(d[1]) + CLASSIFICATION_FOV_RADIUS,
                    i_t
                ] = 1
    return other_agent_fovs


def make_path_fovs(paths, paths_until_col, agent, t_until_col, radius):
    """create for the agent a set of layers indicating their single-agent
    paths."""
    lengths = map(lambda x: x.shape[0], paths)
    path_fovs = init_empty_fov(radius, t_until_col + 1)
    paths_other_agents_fovs = init_empty_fov(radius, t_until_col + 1)
    for i_t_steps in range(t_until_col + 1):
        for i_a in range(len(paths)):
            if i_a == agent:
                fov_to_write = path_fovs
            else:
                fov_to_write = paths_other_agents_fovs
            pos = paths_until_col[agent][i_t_steps]
            for i_t_path in range(paths[i_a].shape[0]):
                d = paths[i_a][i_t_path] - pos
                if (abs(d[0]) <= CLASSIFICATION_FOV_RADIUS and
                        abs(d[1]) <= CLASSIFICATION_FOV_RADIUS):
                    fov_to_write[
                        int(d[0]) + CLASSIFICATION_FOV_RADIUS,
                        int(d[1]) + CLASSIFICATION_FOV_RADIUS,
                        i_t_steps
                    ] = 1
    return path_fovs, paths_other_agents_fovs


def init_empty_fov(radius, t):
    return np.zeros([
        1 + 2 * radius,
        1 + 2 * radius,
        t
    ])


def make_target_deltas(path, t):
    """along the path, construct the deltas between each current position and
    the end of the path until (including) time t."""
    deltas = []
    goal = path[-1][:2]
    for i_t in range(t+1):
        pos = path[i_t]
        deltas.append(goal - pos)
    return deltas


def get_path(path_data, t):
    """from the path data, make a single agent path until
    (and including) time t. (t = -1 gives full path)"""
    path = []
    if t < 0:
        t = path_data.shape[0] - 1
    for i_t in range(t+1):
        if i_t < path_data.shape[0]:
            pos = path_data[i_t][:2]
        else:
            pos = path_data[-1][:2]  # goal
        path.append(pos)
    return np.array(path)


def plot_map_and_paths(gridmap, blocks, data, n_agents):
    """plot the map, agent paths and their blocks.
    You may call `plt.show()` afterwards."""
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    show_map(gridmap)
    block_coords = get_vertex_block_coords(blocks)
    agent_paths = get_agent_paths_from_data(data)
    for ia in range(n_agents):
        plt.plot(agent_paths[ia][:, 0],
                 agent_paths[ia][:, 1],
                 '-',
                 color=colors[ia])
        if block_coords[ia].shape[0]:
            plt.plot(block_coords[ia][:, 0],
                     block_coords[ia][:, 1],
                     'x',
                     color=colors[ia])
    plt.show()


def save_data(data_dict, fname_pkl):
    """save the current data dict in the given pickle file."""
    with open(fname_pkl, 'wb') as f:
        pickle.dump(data_dict, f)


if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', help='mode', choices=(
        TRANSFER_LSTM_STR,
        TRANSFER_CLASSIFICATION_STR,
        GENERATE_SIM_STR))
    parser.add_argument('fname_write_pkl', type=argparse.FileType('wb'))
    parser.add_argument(
        'fname_read_pkl', type=argparse.FileType('rb'), nargs='?')
    args = parser.parse_args()

    if (args.mode == TRANSFER_LSTM_STR or
            args.mode == TRANSFER_CLASSIFICATION_STR):
        # transfer mode lstm
        training_data_we_want = []
        with open(args.fname_read_pkl.name, 'rb') as f:
            all_data = pickle.load(f)
        for d in all_data:
            training_data_we_want.extend(
                training_samples_from_data(d, args.mode))
        save_data(training_data_we_want, args.fname_write_pkl.name)
    elif args.mode == GENERATE_SIM_STR:
        # generation mode
        all_data = []
        plot = False
        width = 10
        height = 10
        n_agents = 8
        n_data_to_gen = 5000
        fill = .4
        # start
        random.seed(0)
        seed = 0
        while len(all_data) < n_data_to_gen:
            collide_count = 0
            while collide_count < 1:
                # gridmap = generate_random_gridmap(width, height, fill)
                # gridmap = initialize_environment(width, fill)
                # starts = [get_random_free_pos(gridmap)
                #           for _ in range(n_agents)]
                # goals = [get_random_free_pos(gridmap)
                #          for _ in range(n_agents)]
                gridmap, starts, goals = tracing_pathes_in_the_dark(
                    width, fill, n_agents, seed
                )
                seed += 1
                collisions, indep_agent_paths = will_they_collide(
                    gridmap, starts, goals)
                collide_count = len(collisions.keys())
            logger.debug(collisions)

            data = plan_in_gridmap(gridmap, starts, goals)

            if data and BLOCKS_STR in data.keys():  # has blocks
                blocks = data[BLOCKS_STR]
                at_least_one_block = has_one_or_more_vertex_blocks(blocks)
                if at_least_one_block:
                    # we take only these for learning
                    data.update({
                        INDEP_AGENT_PATHS_STR: indep_agent_paths,
                        COLLISIONS_STR: collisions,
                        GRIDMAP_STR: gridmap
                    })
                    all_data.append(data)
                    save_data(all_data, args.fname_write_pkl.name)
                    logger.info('Generated {} of {} samples ({}%)'.format(
                        len(all_data),
                        n_data_to_gen,
                        int(100. * len(all_data) / n_data_to_gen)
                    ))
                    if plot:
                        plot_map_and_paths(gridmap, blocks, data, n_agents)
    else:
        raise NotImplementedError(
            "mode >{}< is not implemented yet".format(args.mode))
