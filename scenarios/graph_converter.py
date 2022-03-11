import networkx as nx
import numpy as np
from typing import Tuple, List
from definitions import FREE, POS


def coordinate_to_node(
        env: np.ndarray,
        coordinate: Tuple[int, int]) -> int:
    """Convert coordinates in env to a node number"""
    if coordinate[0] < 0 or coordinate[0] >= env.shape[0]:
        raise IndexError
    if coordinate[1] < 0 or coordinate[1] >= env.shape[1]:
        raise IndexError
    width = env.shape[1]
    return int(coordinate[1] + coordinate[0] * width)


def node_to_coordinate(
        env: np.ndarray,
        node: int) -> Tuple[int, int]:
    """Convert node number to coordinates in env"""
    if node < 0 or node >= env.size:
        raise IndexError
    width = env.shape[1]
    return node // width, node % width


def gridmap_to_nx(env: np.ndarray) -> nx.Graph:
    """Convert a gridmap to a networkx graph"""
    g = nx.Graph()
    pos = {}
    for i, j in np.ndindex(env.shape):
        if env[i, j] == FREE:
            g.add_node(coordinate_to_node(env, (i, j)))
            pos[coordinate_to_node(env, (i, j))] = (float(i), float(j))
        for di, dj in [(-1, 0), (0, -1)]:
            if i+di >= 0 and j+dj >= 0:
                if env[i + di, j + dj] == FREE:
                    g.add_edge(coordinate_to_node(env, (i, j)),
                               coordinate_to_node(env, (i + di, j + dj)))
    nx.set_node_attributes(g, pos, POS)
    return g


def starts_or_goals_to_nodes(
        starts_or_goals: np.ndarray,
        env: np.ndarray) -> List[int]:
    """Convert starts and goals to node numbers"""
    return [
        coordinate_to_node(env, coordinate) for coordinate in starts_or_goals
    ]
