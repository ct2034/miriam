#!/usr/bin/env python3

import logging
import random

import matplotlib.pyplot as plt
import numpy as np

from libMultiRobotPlanning.plan_ecbs import plan_in_gridmap, BLOCKS_STR


VERTEX_CONSTRAINTS_STR = 'vertexConstraints'
SCHEDULE_STR = 'schedule'


def make_random_gridmap(width: int, height: int, fill: float) -> np.ndarray:
    gridmap = np.zeros((width, height))
    obstacle_cells = int(fill * width * height)
    for _ in range(obstacle_cells):
        gridmap[
            random.randint(0, width-1),
            random.randint(0, height-1)
        ] = 1
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


def get_agent_paths_from_data(data):
    agent_paths = []
    if SCHEDULE_STR in data.keys():
        schedule = data[SCHEDULE_STR]
        for agent_str in schedule.keys():
            path_pa = []
            schedule_pa = schedule[agent_str]
            for pose in schedule_pa:
                path_pa.append([
                    pose['x'],
                    pose['y']
                ])
            agent_paths.append(np.array(path_pa))
    return agent_paths


if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    plot = True
    width = 8
    height = 8
    random.seed(1)
    n_agents = 3
    gridmap = make_random_gridmap(width, height, .4)
    count_blocks = 0
    colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k']
    while count_blocks < 2:
        starts = [get_random_free_pos(gridmap, width, height)
                  for _ in range(n_agents)]
        goals = [get_random_free_pos(gridmap, width, height)
                 for _ in range(n_agents)]
        data = plan_in_gridmap(gridmap, starts, goals)

        if data and BLOCKS_STR in data.keys():
            blocks = data[BLOCKS_STR]
            has_blocks = not all(v == 0 for v in blocks.values())
            if has_blocks:
                count_blocks += 1
                logger.info("blocks:" + str(blocks))
                if plot:
                    show_map(gridmap)
                    block_coords = get_vertex_block_coords(blocks)
                    agent_paths = get_agent_paths_from_data(data)
                    for ia in range(len(agent_paths)):
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
