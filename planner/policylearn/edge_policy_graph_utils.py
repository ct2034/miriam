import networkx as nx
import torch
from definitions import POS
from torch_geometric.data import Data


def agents_to_data(agents, i_self):
    HOP_DIST = 3
    own_pos = agents[i_self].pos
    assert agents[i_self].has_roadmap
    g = agents[i_self].env
    g_sml = nx.ego_graph(g, own_pos, radius=HOP_DIST)
    big_from_small = {i: n for i, n in enumerate(g_sml.nodes)}
    small_from_big = {n: i for i, n in big_from_small.items()}
    pos = nx.get_node_attributes(g, POS)
    # data layers
    x_layer_own_path = torch.zeros((len(small_from_big), 1))
    for p, t in agents[i_self].path:
        if p in g_sml.nodes:
            p_sml = small_from_big[p]
            x_layer_own_path[p_sml] = t
    x_layer_other_paths = torch.zeros((len(small_from_big), 1))
    for i_a, a in enumerate(agents):
        if i_a == i_self:
            continue
        for p, t in a.path:
            if p in g_sml.nodes:
                p_sml = small_from_big[p]
                x_layer_other_paths[p_sml] = max(x_layer_other_paths[p_sml], t)
    d = Data(
        pos=torch.tensor([pos[n] for n in g_sml.nodes]),
        edge_index=torch.tensor([(
            small_from_big[n1],
            small_from_big[n2]
        ) for (n1, n2) in g_sml.edges]).t(),
        x=torch.cat([x_layer_own_path, x_layer_other_paths], dim=1)
    )
    return d, small_from_big[own_pos]
