from math import atan2, cos, pi, sin

import networkx as nx
import numpy as np
import scenarios
import torch
from definitions import INVALID, POS
from torch_geometric.data import Data

RADIUS = .001
TIMEOUT = 120


def t_to_data(t: int, path_i: int) -> float:
    """Convert a time step to a data layer info.
    Such that the datapoint at the current pose is 1.0,
    and increases by .1 every step towards the goal.
    And zero for all past points."""
    if t < path_i:
        return 0.
    else:
        return 1. + (t - path_i) * .1


def agents_to_data(agents, i_self: int, hop_dist: int = 3) -> Data:
    # preparation
    self_node = agents[i_self].pos
    assert agents[i_self].has_roadmap
    g = agents[i_self].env
    g_sml = nx.ego_graph(g, self_node, radius=hop_dist)
    # for a index counted in the subgraph, get index in full graph
    big_from_small = {i: int(n) for i, n in enumerate(g_sml.nodes)}
    # for a node in the full graph, get index in subgraph (if it is in there)
    small_from_big = {n: i for i, n in big_from_small.items()}
    pos = nx.get_node_attributes(g, POS)
    self_pos = pos[self_node]
    self_next_pos = pos[agents[i_self].what_is_next_step()]
    if self_next_pos is None or (self_next_pos == self_pos).all():
        self_heading = 0.  # TODO: better idea?
    else:
        self_heading = atan2(
            self_next_pos[1] - self_pos[1],
            self_next_pos[0] - self_pos[0]) - pi / 2  # zero is up

    # data layers
    # * relative node positions
    rot = np.array([[cos(-self_heading), -sin(-self_heading)],
                    [sin(-self_heading), cos(-self_heading)]])
    x_layer_relative_pos = torch.zeros(len(g_sml.nodes), 2)
    for i, n in enumerate(g_sml.nodes):
        x_layer_relative_pos[i] = torch.tensor(
            rot @ (np.array(pos[n]) - np.array(self_pos)))

    # * relative node distances
    x_layer_relative_dist = torch.norm(x_layer_relative_pos, dim=1)
    x_layer_relative_dist.resize_(x_layer_relative_dist.shape[0], 1)

    # * relative node angles
    x_layer_relative_angle = torch.atan2(
        x_layer_relative_pos[:, 1], x_layer_relative_pos[:, 0]) - pi / 2
    x_layer_relative_angle[x_layer_relative_pos[:, 0] == 0] = 0
    x_layer_relative_angle.resize_(x_layer_relative_angle.shape[0], 1)

    # * own path
    x_layer_own_path = torch.zeros((len(small_from_big), 1))
    for p, t in agents[i_self].path:
        if p in g_sml.nodes:
            p_sml = small_from_big[p]
            x_layer_own_path[p_sml] = t_to_data(t, agents[i_self].path_i)
    # * other paths
    x_layer_other_paths = torch.zeros((len(small_from_big), 1))
    for i_a, a in enumerate(agents):
        if i_a == i_self:
            continue
        for p, t in a.path:
            if p not in g_sml.nodes:
                continue
            p_sml = small_from_big[p]
            x_layer_other_paths[p_sml] = max(
                x_layer_other_paths[p_sml],
                t_to_data(t, a.path_i))
    # * other poses
    x_layer_other_poses = torch.zeros((len(small_from_big), 1))
    for i_a, a in enumerate(agents):
        if i_a == i_self:
            continue
        if a.pos not in g_sml.nodes:
            continue
        p_sml = small_from_big[a.pos]
        x_layer_other_poses[p_sml] = 1.

    d = Data(
        pos=torch.tensor([pos[n] for n in g_sml.nodes]),
        edge_index=torch.tensor([(
            small_from_big[n1],
            small_from_big[n2]
        ) for (n1, n2) in g_sml.edges]).t(),
        x=torch.cat([x_layer_relative_pos,
                     x_layer_relative_dist,
                     x_layer_relative_angle,
                     x_layer_own_path,
                     x_layer_other_paths,
                     x_layer_other_poses],
                    dim=1)
    )
    return d, small_from_big[self_node], big_from_small


def get_optimal_edge(agents, i_agent_to_consider):
    """Return the optimal edge to take for the given agent. """
    starts = [a.pos for a in agents]
    goals = [a.goal for a in agents]
    paths = scenarios.solvers.cached_cbsr(
        agents[0].env, starts, goals, radius=RADIUS, timeout=TIMEOUT)
    if paths is INVALID:
        raise RuntimeError("No paths found")
    else:
        path = paths[i_agent_to_consider]
        if len(path) == 1:  # already at goal
            return path[0][0]
        return path[1][0]
