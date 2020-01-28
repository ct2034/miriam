#!/usr/bin/env python3

import argparse
import logging
import pickle
import random
import sys

import matplotlib.pyplot as plt
import numpy as np

from libMultiRobotPlanning.plan_ecbs import BLOCKS_STR, plan_in_gridmap

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

LSTM_FOV_RADIUS = 2  # self plus x in all 4 dircetions
CLASSIFICATION_POS_TIMESTEPS = 3
CLASSIFICATION_FOV_RADIUS = 3
DTYPE_SAMPLES = np.int8


def generate_random_gridmap(width: int, height: int, fill: float):
    gridmap = np.zeros((width, height), dtype=DTYPE_SAMPLES)
    while np.count_nonzero(gridmap) < fill * width * height:
        direction = random.randint(0, 1)
        start = (
            random.randint(0, width-1),
            random.randint(0, height-1)
        )
        if direction:  # x
            gridmap[start[0]:random.randint(0, width-1), start[1]] = 1
        else:  # y
            gridmap[start[0], start[1]:random.randint(0, height-1)] = 1
    return gridmap


def show_map(x):
    plt.imshow(
        np.swapaxes(x, 0, 1),
        aspect='equal',
        cmap=plt.get_cmap("binary"),
        origin='lower')


def is_free(gridmap, pos):
    return gridmap[tuple(pos)] == 0


def get_random_free_pos(gridmap, width, height):
    def random_pos(width, height):
        return [
            random.randint(0, width-1),
            random.randint(0, height-1)
        ]
    pos = random_pos(width, height)
    while not is_free(gridmap, pos):
        pos = random_pos(width, height)
    return pos


def get_vertex_block_coords(blocks):  # TODO: edge constraints
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


def has_exatly_one_vertex_block(blocks):
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
    return n_ec == 0 and n_vc == 1


def get_agent_paths_from_data(data, timed=False):
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


def will_they_collide(gridmap, starts, goals):
    collisions = {}
    seen = set()
    been_at = {}
    agent_paths = []
    for i_a, _ in enumerate(starts):
        data = plan_in_gridmap(gridmap, [starts[i_a], ], [
                               goals[i_a], ], timeout=2)
        if data is None:
            return {}, []
        single_agent_paths = get_agent_paths_from_data(data, timed=True)
        if not single_agent_paths:
            return {}, []
        agent_paths.append(single_agent_paths[0])
        for pos in single_agent_paths[0]:
            pos = tuple(pos)
            if pos in seen:  # TODO: edge collisions
                collisions[pos] = (been_at[pos], i_a)
            seen.add(pos)
            been_at[pos] = i_a
    return collisions, agent_paths


def add_padding_to_gridmap(gridmap, radius):
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
    training_samples = []
    n_agents = len(data[INDEP_AGENT_PATHS_STR])
    assert len(data[COLLISIONS_STR]
               ) == 1, "assuming we only handle one conflict"
    for col_vertex, col_agents in data[COLLISIONS_STR].items():
        t = col_vertex[2]
        blocked_agent = -1
        unblocked_agent = -1
        for i_a in col_agents:
            if data[BLOCKS_STR]["agent"+str(i_a)] is dict:
                blocked_agent = i_a
            else:
                unblocked_agent = i_a
        data_pa = []
        if mode == TRANSFER_LSTM_STR:
            training_samples.extend(lstm_samples(
                n_agents, data, t, data_pa, col_agents, unblocked_agent))
        elif mode == TRANSFER_CLASSIFICATION_STR:
            training_samples.extend(classification_samples(
                n_agents, data, t, data_pa, col_agents, unblocked_agent))
    return training_samples


def lstm_samples(n_agents, data, t, data_pa, col_agents, unblocked_agent):
    training_samples = []
    padded_gridmap = add_padding_to_gridmap(data[GRIDMAP_STR], LSTM_FOV_RADIUS)
    for i_a in range(n_agents):
        # training features
        path = make_path(data[INDEP_AGENT_PATHS_STR][i_a], t)
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


def classification_samples(n_agents, data, t, data_pa, col_agents,
                           unblocked_agent):
    training_samples = []
    paths = []
    padded_gridmap = add_padding_to_gridmap(data[GRIDMAP_STR],
                                            CLASSIFICATION_FOV_RADIUS)
    for i_a in range(n_agents):
        path = make_path(data[INDEP_AGENT_PATHS_STR][i_a], t)
        if len(path) < CLASSIFICATION_POS_TIMESTEPS:
            padded_path = np.zeros([CLASSIFICATION_POS_TIMESTEPS, 2])
            for i in range(0, CLASSIFICATION_POS_TIMESTEPS - len(path)):
                padded_path[i] = path[0]
            padded_path[i+1:] = path
            path = padded_path
        paths.append(path[-CLASSIFICATION_POS_TIMESTEPS:])
    t = CLASSIFICATION_POS_TIMESTEPS-1
    for i_a in col_agents:
        obstacle_fovs = make_obstacle_fovs(padded_gridmap,
                                           paths[i_a],
                                           t,
                                           CLASSIFICATION_FOV_RADIUS)
        pos_fovs = make_pos_fovs(paths, i_a, CLASSIFICATION_FOV_RADIUS)
        np_obstacle_fovs = np.stack(obstacle_fovs, axis=2)
        x = np.append(np_obstacle_fovs, pos_fovs, axis=2)
        training_samples.append((
            x,
            1 if i_a == unblocked_agent else 0))
    return training_samples


def make_obstacle_fovs(padded_gridmap, path, t, radius):
    """create for all agents a set of FOVS of radius contatining positions of
    obstacles in gridmap."""
    fovs = []
    for i_t in range(t+1):
        pos = path[i_t]
        fovs.append(
            padded_gridmap[
                int(pos[0]):int(pos[0]) + 1 + 2 * radius,
                int(pos[1]):int(pos[1]) + 1 + 2 * radius
            ]
        )
    return fovs


def make_pos_fovs(paths, agent, radius):
    """create for the agent a set of FOVS of radius contatining positions of
    other agents."""
    t = paths[0].shape[0]
    pos_fovs = np.zeros([
        1 + 2 * radius,
        1 + 2 * radius,
        t
    ])
    for i_t in range(t):
        pos = paths[agent][i_t]
        for i_a in [i for i in range(len(paths)) if i != agent]:
            d = paths[i_a][i_t] - pos
            if (abs(d[0]) <= CLASSIFICATION_FOV_RADIUS and
                    abs(d[1]) <= CLASSIFICATION_FOV_RADIUS):
                pos_fovs[
                    int(d[0]) + CLASSIFICATION_FOV_RADIUS,
                    int(d[1]) + CLASSIFICATION_FOV_RADIUS,
                    i_t
                ] = 1
    return pos_fovs


def make_target_deltas(path, t):
    deltas = []
    goal = path[-1][:2]
    for i_t in range(t+1):
        pos = path[i_t]
        deltas.append(goal - pos)
    return deltas


def make_path(path_data, t):
    path = []
    for i_t in range(t+1):
        if t < path_data.shape[0]:
            pos = path_data[i_t][:2]
        else:
            pos = path_data[-1][:2]  # goal
        path.append(pos)
    return np.array(path)


def plot_map_and_paths(gridmap, blocks, data, n_agents):
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
        random.seed(0)
        n_agents = 5
        n_data_to_gen = 5000
        while len(all_data) < n_data_to_gen:
            collide_count = 0
            while collide_count != 1:
                gridmap = generate_random_gridmap(width, height, .2)
                starts = [get_random_free_pos(gridmap, width, height)
                          for _ in range(n_agents)]
                goals = [get_random_free_pos(gridmap, width, height)
                         for _ in range(n_agents)]
                collisions, indep_agent_paths = will_they_collide(
                    gridmap, starts, goals)
                collide_count = len(collisions.keys())
            logger.debug(collisions)

            data = plan_in_gridmap(gridmap, starts, goals)

            if data and BLOCKS_STR in data.keys():
                blocks = data[BLOCKS_STR]
                has_a_block = has_exatly_one_vertex_block(blocks)
                if has_a_block:
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
