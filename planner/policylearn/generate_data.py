#!/usr/bin/env python3

import logging
import random

import matplotlib.pyplot as plt
import numpy as np

from libMultiRobotPlanning.plan_ecbs import plan_in_gridmap, BLOCKS_STR


VERTEX_CONSTRAINTS_STR = 'vertexConstraints'
EDGE_CONSTRAINTS_STR = 'edgeConstraints'
SCHEDULE_STR = 'schedule'
INDEP_AGENT_PATHS_STR = 'indepAgentPaths'
COLLISIONS_STR = 'collisions'
GRIDMAP_STR = 'gridmap'
OWN_STR = 'own'
OTHERS_STR = 'others'

FOV_RADIUS = 2  # self plus x in all 4 dircetions
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


def add_padding_to_gridmap(gridmap):
    size = gridmap.shape
    padded_gridmap = np.ones([
        size[0] + 2 * FOV_RADIUS,
        size[1] + 2 * FOV_RADIUS],
        dtype=DTYPE_SAMPLES)
    padded_gridmap[
        FOV_RADIUS:size[0]+FOV_RADIUS,
        FOV_RADIUS:size[1]+FOV_RADIUS] = gridmap
    return padded_gridmap


def training_samples_from_data(data):
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
        for i_a in range(n_agents):
            # training features
            fovs = make_fovs(data[GRIDMAP_STR],
                             data[INDEP_AGENT_PATHS_STR][i_a],
                             t)
            deltas = make_target_deltas(data[INDEP_AGENT_PATHS_STR][i_a],
                                        t)
            data_this_agent = []
            for i_t in range(t+1):
                data_this_agent.append(
                    np.append(
                        np.concatenate(fovs[i_t]),
                        [
                            deltas[i_t]
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


def make_fovs(gridmap, path, t):
    fovs = []
    for i_t in range(t+1):
        if t < path.shape[0]:
            pos = path[i_t][:2]
        else:
            pos = path[-1][:2]
        fovs.append(
            gridmap[
                pos[0]:pos[0] + 1 + 2 * FOV_RADIUS,
                pos[1]:pos[1] + 1 + 2 * FOV_RADIUS
            ]
        )
    return fovs


def make_target_deltas(path, t):
    assert len(path) >= t, "t must be in path"
    deltas = []
    goal = path[-1][:2]
    for i_t in range(t+1):
        if t < path.shape[0]:
            pos = path[i_t][:2]
        else:
            pos = path[-1][:2]
        deltas.append(goal - pos)
    return deltas


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


if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    training_data_we_want = []
    plot = False
    width = 10
    height = 10
    random.seed(0)
    n_agents = 5
    n_data_to_gen = 3
    while len(training_data_we_want) < n_data_to_gen:
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
                    GRIDMAP_STR: add_padding_to_gridmap(gridmap)
                })
                training_data_we_want.extend(
                    training_samples_from_data(data)
                )
                logger.info("blocks:" + str(blocks))
                if plot:
                    plot_map_and_paths(gridmap, blocks, data, n_agents)
    print(training_data_we_want)
