from itertools import product

import numpy as np
import torch
from definitions import FREE
from torch_geometric.data import Data


def node_to_pos(data_pos, n):
    return data_pos[n, :]


def pos_to_node(data_pos, pos):
    n_nodes = data_pos.shape[0]
    for i in range(n_nodes):
        if data_pos[i, 0] == pos[1] and data_pos[i, 1] == pos[0]:
            return i
    return None


def gridmap_to_graph(gridmap):
    width, height = gridmap.shape
    nodes = []
    edges = []
    for x, y in product(range(width), range(height)):
        node = (x, y)
        if gridmap[y, x] == FREE:
            nodes.append(node)
            if x > 0:
                left_n = (x - 1, y)
                if left_n in nodes:
                    edges.append([
                        nodes.index(node),
                        nodes.index(left_n)
                    ])
            if y > 0:
                above_n = (x, y - 1)
                if above_n in nodes:
                    edges.append([
                        nodes.index(node),
                        nodes.index(above_n)
                    ])
    data_edge_index = torch.tensor(edges).T
    data_pos = torch.tensor(nodes)
    assert(data_edge_index.shape[0] == 2)
    assert(data_edge_index.shape[1] == len(edges))
    assert(data_pos.shape[0] == len(nodes))
    assert(data_pos.shape[1] == 2)
    return data_edge_index, data_pos


def get_agent_pos_layer(data_pos, paths_until_col, i_a):
    n_nodes = data_pos.shape[0]
    data_x_slice = torch.zeros((n_nodes, 1))
    pos = paths_until_col[i_a][0]
    node = pos_to_node(data_pos, pos)
    assert node is not None
    data_x_slice[node, 0] = 1
    return data_x_slice


def get_agent_path_layer(data_pos, paths_full, i_a):
    n_nodes = data_pos.shape[0]
    data_x_slice = torch.zeros((n_nodes, 1))
    for pos in paths_full[i_a]:
        node = pos_to_node(data_pos, pos)
        assert node is not None
        data_x_slice[node, 0] = 1
    return data_x_slice


def to_data_obj(data_x, data_edge_index, data_pos, y):
    return Data(
        x=data_x,
        edge_index=data_edge_index,
        pos=data_pos,
        y=y
    )
