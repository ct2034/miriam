#!/usr/bin/env python3
import argparse
import datetime
import logging
import math
import os
import pickle
import random
import uuid
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
from definitions import FREE, INVALID
from planner.policylearn.generate_fovs import *
from planner.policylearn.libMultiRobotPlanning.plan_ecbs import BLOCKS_STR
from scenarios.evaluators import cached_ecbs
from scenarios.generators import tracing_pathes_in_the_dark
from sim.decentralized.agent import Agent
from tools import ProgressBar

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
NO_SOLUTION_STR = 'no_solution'

LSTM_FOV_RADIUS = 2  # self plus x in all 4 directions
CLASSIFICATION_POS_TIMESTEPS = 3
CLASSIFICATION_FOV_RADIUS = 3  # self plus x in all 4 directions
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
        if blocks[agent] != 0 and blocks[agent] is not None:
            blocks_pa = blocks[agent]
            if (VERTEX_CONSTRAINTS_STR in blocks_pa.keys() and
                    blocks_pa[VERTEX_CONSTRAINTS_STR] is not None):
                for _ in blocks_pa[VERTEX_CONSTRAINTS_STR]:
                    n_vc += 1
            if (EDGE_CONSTRAINTS_STR in blocks_pa.keys() and
                    blocks_pa[EDGE_CONSTRAINTS_STR] is not None):
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
    if blocks is None:
        return False
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


def time_path(path: np.ndarray):
    nrs = np.array([np.arange(path.shape[0])]).transpose()
    return np.append(path, nrs, axis=1)


def will_they_collide(gridmap, starts, goals):
    """checks if for a given set of starts and goals the agents travelling
    between may collide on the given gridmap."""
    do_collide = False
    seen = set()
    agent_paths = []
    for i_a, _ in enumerate(starts):
        a = Agent(gridmap, starts[i_a])
        success = a.give_a_goal(goals[i_a])
        tp = time_path(a.path)
        agent_paths.append(tp)
        for pos in tp:
            if not do_collide:  # only need to do this if no collision was found
                if tuple(pos) in seen:
                    do_collide = True
                seen.add(tuple(pos))
    return do_collide, agent_paths


def where_will_they_collide(agent_paths, starts):
    """checks if for a given set of starts and goals the agents travelling
    between may collide on the given gridmap."""
    collisions = {}
    seen_pos = set()
    seen_edge = set()
    been_at = {}
    for i_a, _ in enumerate(starts):
        prev_pos = None
        for pos in agent_paths[i_a]:
            pos = tuple(pos)
            # vertex collisions
            if pos in seen_pos:
                collisions[pos] = (been_at[pos], i_a)
            seen_pos.add(pos)
            been_at[pos] = i_a
            # edge collisions
            if prev_pos is not None:
                edge = tuple(sorted((prev_pos, pos)))
                if edge in seen_edge:
                    collisions[edge] = (been_at[edge], i_a)
                seen_edge.add(edge)
                been_at[edge] = i_a
            prev_pos = pos
    return collisions


def is_vertex_coll(collision):
    if len(collision) == 2:
        assert len(collision[0]) == 3
        return False  # edge
    else:
        assert isinstance(collision[0], (int, np.int64))
        return True  # vertex


def get_other(agents: tuple, agent):
    assert agent in agents
    assert len(agents) == 2
    if agent == agents[0]:
        return agents[1]
    else:
        return agents[0]


def blocks_from_data_to_lists(data):
    data_blocks = data[BLOCKS_STR]
    n_agents = len(data_blocks)
    blocks = [None] * n_agents
    for agent_str, data_blocks_pa in data_blocks.items():
        i_a = int(agent_str.replace("agent", ""))
        if data_blocks_pa != 0:
            blocks_pa = []
            if VERTEX_CONSTRAINTS_STR in data_blocks_pa.keys():
                for db in data_blocks_pa[VERTEX_CONSTRAINTS_STR]:
                    block = (
                        db['v.x'],
                        db['v.y'],
                        db['t']
                    )
                    blocks_pa.append(block)
            if EDGE_CONSTRAINTS_STR in data_blocks_pa.keys():
                for db in data_blocks_pa[EDGE_CONSTRAINTS_STR]:
                    block = ((
                        db['v1.x'],
                        db['v1.y'],
                        db['t']
                    ), (
                        db['v2.x'],
                        db['v2.y'],
                        db['t']+1
                    ))
                    blocks_pa.append(block)
            blocks[i_a] = blocks_pa
    return blocks


def is_collision_in_blocks(collision, blocks):
    if blocks is None:
        return False
    for b in blocks:
        if len(b) == 3 and len(collision) == 3:  # vertex
            if collision == b:
                return True
        elif len(b) == 2 and len(collision) == 2:  # edge
            if collision[0] == b[0] or collision[1] == b[1]:
                return True
        elif len(b) == 2 and len(collision) == 3:  # vertex coll, edge block
            if collision == b[0] or collision == b[1]:
                return True
    return False


def training_samples_from_data(data, mode):
    """extract training samples from the data simulation data dict."""
    training_samples = []
    paths = data[INDEP_AGENT_PATHS_STR]
    n_agents = len(paths)
    bs = blocks_from_data_to_lists(data)
    for collision, col_agents in data[COLLISIONS_STR].items():
        unblocked_agent = None
        for i_a in col_agents:
            i_oa = get_other(col_agents, i_a)
            if is_vertex_coll(collision):
                pos = collision
            else:
                pos = collision[0]
            t = pos[2]
            if pos in paths[i_a] or collision in paths[i_oa]:
                if is_collision_in_blocks(collision, bs[i_a]):
                    unblocked_agent = i_a
                elif is_collision_in_blocks(collision, bs[i_oa]):
                    unblocked_agent = i_oa
        if unblocked_agent is not None:  # if we were able to find it
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
    assert len(col_agents) == 2, "assuming two agent in colission here"
    for i_ca, i_a in enumerate(col_agents):
        i_oa = col_agents[(i_ca+1) % 2]
        x = extract_all_fovs(t, paths_until_col, paths_full,
                             padded_gridmap, i_a, i_oa, CLASSIFICATION_FOV_RADIUS)
        training_samples.append((
            x,
            1 if i_a == unblocked_agent else 0))
    return training_samples


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


def insert_str_before_extension(fname: str, to_insert: str):
    """in a filename, insert string before extension"""
    parts = fname.split(".")
    parts[0] = parts[0] + to_insert
    return ".".join(parts)


def save_data(data_dict, fname_pkl):
    """save the current data dict in the given pickle file."""
    with open(fname_pkl, 'wb') as f:
        pickle.dump(data_dict, f)


def simulate_one_data(width, fill, n_agents, base_seed):
    data_ok = False
    random.seed(base_seed)
    seed = base_seed
    while not data_ok:
        do_collide = False
        while not do_collide:
            gridmap, starts, goals = tracing_pathes_in_the_dark(
                width, fill, n_agents, seed
            )
            seed += random.randint(0, 10E6)
            do_collide, indep_agent_paths = will_they_collide(
                gridmap, starts, goals)

        data = cached_ecbs(
            gridmap, starts, goals, timeout=10)
        collisions = where_will_they_collide(indep_agent_paths, starts)

        data_ok = (
            data != INVALID and
            BLOCKS_STR in data.keys() and
            has_one_or_more_vertex_blocks(data[BLOCKS_STR])
        )
        if data_ok:
            # we take only these for learning
            data.update({
                INDEP_AGENT_PATHS_STR: indep_agent_paths,
                COLLISIONS_STR: collisions,
                GRIDMAP_STR: gridmap,
                BLOCKS_STR: data[BLOCKS_STR]
            })
    return data


if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', help='mode', choices=(
        TRANSFER_LSTM_STR,
        TRANSFER_CLASSIFICATION_STR,
        GENERATE_SIM_STR,
        NO_SOLUTION_STR))
    parser.add_argument('fname_write_pkl',
                        type=argparse.FileType('wb'), nargs='?')
    parser.add_argument(
        'fname_read_pkl', type=argparse.FileType('rb'), nargs='?')
    args = parser.parse_args()

    start_time = datetime.datetime.now()

    if (args.mode == TRANSFER_LSTM_STR or
            args.mode == TRANSFER_CLASSIFICATION_STR):
        # transfer mode lstm
        training_data_we_want = []
        with open(args.fname_read_pkl.name, 'rb') as f:
            all_data = pickle.load(f)
        data_len = len(all_data)
        i = 0
        pb = ProgressBar("main", data_len, 1)
        for d in all_data:
            i += 1
            pb.progress()
            training_data_we_want.extend(
                training_samples_from_data(d, args.mode))
        save_data(training_data_we_want, args.fname_write_pkl.name)
        print(f'got {len(training_data_we_want)} samples '
              + f'from {len(all_data)} simulations')
        pb.end()
    elif args.mode == GENERATE_SIM_STR:
        # scenario paramters
        width = 8
        height = width
        n_agents = 8
        fill = .4
        # generation parameters
        plot = False
        n_data_to_gen = int(os.getenv("N_DATA_TO_GEN", 5000))
        batch_size = int(n_data_to_gen / 10)
        n_batches = math.ceil(n_data_to_gen / batch_size)
        logger.info(f'Generating {n_data_to_gen} data points ' +
                    f'in {n_batches} batches of {batch_size}.')
        seed = os.getenv("SEED", 0)
        random.seed(seed)
        logger.info(f"Using initial seed: {seed}")
        # start
        with Pool(4) as p:
            pb_main = ProgressBar(f'main', n_batches)
            for i_batch in range(n_batches):
                start = batch_size * i_batch
                stop = min(start + batch_size, n_data_to_gen)
                arguments = [(width, fill, n_agents, seed)
                             for seed in range(start, stop)]
                batch_data = p.starmap(simulate_one_data, arguments)
                save_data(batch_data, insert_str_before_extension(args.fname_write_pkl.name, f'{i_batch:02}'))
                pb_main.progress()
            # save in the end for sure
            pb_main.end()
    elif args.mode == NO_SOLUTION_STR:
        # generate data of scenarios without solution (no info on how to solve
        # collision) for autoencoding.
        # scenario paramters
        width = 8
        height = width
        n_agents = 8
        fill = .4
        # generation parameters
        plot = False
        all_data = []
        n_data_to_gen = int(os.getenv("N_DATA_TO_GEN", 2 ** 10))
        logger.info("Generating {} data points.".format(n_data_to_gen))
        # seed = int(os.getenv("SEED", 0))
        my_uuid = uuid.uuid4()
        logger.info("My UUID: {}".format(my_uuid))
        seed = my_uuid.time
        random.seed(seed)
        logger.info("Using initial seed: {}".format(seed))
        if not args.fname_write_pkl:
            fname = "/data/" + str(my_uuid) + ".pkl"
            logger.warn("No filename provided.")
        else:
            fname = args.fname_write_pkl.name
        logger.warn("Our filename will be: {}".format(fname))
        # start
        while len(all_data) < n_data_to_gen:
            do_collide = False
            while not do_collide:
                gridmap, starts, goals = tracing_pathes_in_the_dark(
                    width, fill, n_agents, seed
                )
                seed += 1
                do_collide, indep_agent_paths = will_they_collide(
                    gridmap, starts, goals)

            data = {
                INDEP_AGENT_PATHS_STR: indep_agent_paths,
                GRIDMAP_STR: gridmap
            }
            all_data.append(data)
            if len(all_data) % 100 == 0:
                save_data(all_data, fname)
            if plot:
                plot_map_and_paths(gridmap, {}, data, n_agents)

    else:
        raise NotImplementedError(
            "mode >{}< is not implemented yet".format(args.mode))
