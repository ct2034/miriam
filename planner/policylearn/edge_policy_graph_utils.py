import networkx as nx
import scenarios
import torch
from definitions import INVALID, POS
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

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


def agents_to_data(agents, i_self: int, hop_dist: int = 3):
    own_node = agents[i_self].pos
    assert agents[i_self].has_roadmap
    g = agents[i_self].env
    g_sml = nx.ego_graph(g, own_node, radius=hop_dist)
    big_from_small = {i: int(n) for i, n in enumerate(g_sml.nodes)}
    small_from_big = {n: i for i, n in big_from_small.items()}
    pos = nx.get_node_attributes(g, POS)
    own_pos = torch.tensor(pos[own_node])
    own_angle = 0.
    try:
        next_node = agents[i_self].path[agents[i_self].path_i + 1][0]
        next_pos = torch.tensor(pos[next_node])
        own_angle = torch.atan2(
            next_pos[1] - own_pos[1], next_pos[0] - own_pos[0]).item()
    except IndexError:
        pass

    # data layers
    # 1. own path
    x_layer_own_path = torch.zeros((len(small_from_big), 1))
    for p, t in agents[i_self].path:
        if p in g_sml.nodes:
            p_sml = small_from_big[p]
            x_layer_own_path[p_sml] = t_to_data(t, agents[i_self].path_i)

    # 2. other paths
    x_layer_other_paths = torch.zeros((len(small_from_big), 1))
    for i_a, a in enumerate(agents):
        if i_a == i_self:
            continue
        for p, t in a.path:
            if p in g_sml.nodes:
                p_sml = small_from_big[p]
                x_layer_other_paths[p_sml] = max(
                    float(x_layer_other_paths[p_sml].item()),
                    t_to_data(t, a.path_i))

    # 3. relative distance
    relative_pos = torch.zeros((len(small_from_big), 2))
    for i_big,  i_sml in small_from_big.items():
        relative_pos[i_sml] = torch.tensor(pos[i_big]) - own_pos
    relative_distance = torch.norm(relative_pos, dim=1)

    # 4. relative angle
    relative_angle = torch.atan2(
        relative_pos[:, 1], relative_pos[:, 0])
    # if not own node, add own angle
    relative_angle[relative_distance != 0] -= own_angle

    # edge index
    edge_index = torch.tensor([(
        small_from_big[n1],
        small_from_big[n2]
    ) for (n1, n2) in g_sml.edges]).t()
    edge_index = to_undirected(edge_index)

    # get y
    node_to_go = small_from_big[get_optimal_edge(agents, i_self)]
    assert torch.tensor([node_to_go, small_from_big[own_node]]) in edge_index.T
    y = torch.zeros(len(small_from_big), dtype=torch.float32)
    y[node_to_go] = 1.

    d = Data(
        edge_index=edge_index,
        x=torch.cat([x_layer_own_path,
                     x_layer_other_paths,
                     relative_distance.view(-1, 1),
                     relative_angle.view(-1, 1)], dim=1),
        y=y
    )
    return d, big_from_small


def get_optimal_edge(agents, i_agent: int):
    """Return the optimal edge to take for the given agent. """
    starts = [a.pos for a in agents]
    goals = [a.goal for a in agents]
    import scenarios.solvers
    paths = scenarios.solvers.cached_cbsr(
        agents[0].env, starts, goals, radius=RADIUS, timeout=TIMEOUT)
    if paths is INVALID:
        raise RuntimeError("No paths found")
    else:
        path = paths[i_agent]
        if len(path) == 1:  # already at goal
            return path[0][0]
        return path[1][0]
