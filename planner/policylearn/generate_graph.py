from itertools import product
from typing import Tuple

import networkx as nx
import numpy as np
import torch
from sortedcontainers import SortedSet
from torch_geometric.data import Data

from definitions import FREE, C


def node_to_pos(data_pos, n):
    return data_pos[n, :]


def pos_to_node(data_pos, pos):
    n_nodes = data_pos.shape[0]
    for i in range(n_nodes):
        if data_pos[i, 0] == pos[0] and data_pos[i, 1] == pos[1]:
            return i
    return None


def gridmap_to_graph(gridmap: np.ndarray, hop_dist: int, own_pos: C):
    width, height = gridmap.shape
    g = nx.grid_2d_graph(width, height)
    g = g.to_undirected(as_view=True)
    if hop_dist < np.inf:
        if isinstance(own_pos, int):
            own_pos = (own_pos,)  # type: ignore
        g = nx.ego_graph(g, own_pos, radius=hop_dist)

    def filter_node(n):
        return gridmap[n] == FREE

    g = nx.subgraph_view(g, filter_node=filter_node)
    nodes = list(g.nodes)
    edges = []
    for a, b in g.edges:
        edges.append((nodes.index(a), nodes.index(b)))
    data_edge_index = torch.tensor(edges).T
    data_pos = torch.tensor(nodes)
    assert data_edge_index.shape[0] == 2
    assert data_edge_index.shape[1] == len(edges)
    assert data_pos.shape[0] == len(nodes)
    assert data_pos.shape[1] == 2
    return data_edge_index, data_pos


def get_agent_pos_layer(data_pos, positions, i_as):
    n_nodes = data_pos.shape[0]
    data_x_slice = torch.zeros((n_nodes, 1))
    for i_a in i_as:
        pos = positions[i_a]
        node = pos_to_node(data_pos, pos)
        if node is not None:
            data_x_slice[node, 0] = 1
    return data_x_slice


def get_agent_path_layer(data_pos, paths, i_as):
    n_nodes = data_pos.shape[0]
    data_x_slice = torch.zeros((n_nodes, 1))
    for i_a in i_as:
        path = paths[i_a]
        previous = -1
        for i_t, pos in enumerate(path):
            node = pos_to_node(data_pos, pos)
            if node is not None and node != previous:
                data_x_slice[node, 0] += (float(i_t) + 1) / len(path)
            previous = node
    return data_x_slice


def to_data_obj(data_x, data_edge_index, data_pos, y):
    return Data(x=data_x, edge_index=data_edge_index, pos=data_pos, y=y)
