import unittest
from itertools import product
from math import atan2, pi
from random import Random

import networkx as nx
import numpy as np
from planner.policylearn.edge_policy_graph_utils import *
from roadmaps.var_odrm_torch.var_odrm_torch import (make_graph_and_flann,
                                                    read_map, sample_points)
from scenarios.test_helper import make_cache_folder_and_set_envvar
from sim.decentralized.agent import Agent
from sim.decentralized.runner import to_agent_objects

def demo():
    n_nodes = 16
    n_smpls = 10
    n_agents = 2
    rng = Random(0)
    map_fname: str = "roadmaps/odrm/odrm_eval/maps/plain.png"
    map_img = read_map(map_fname)
    for _ in range(n_smpls):
        pos = sample_points(n_nodes, map_img, rng)
        g: nx.Graph
        (g, _) = make_graph_and_flann(pos, map_img)
        starts = rng.sample(range(g.number_of_nodes()), k=n_agents)
        goals = rng.sample(range(g.number_of_nodes()), k=n_agents)
        agents = to_agent_objects(g, starts, goals, radius=.1)
        assert agents is not None
        try:
            data, big_from_small = agents_to_data(agents, 0)
        except RuntimeError:
            break
        for n in range(data.num_nodes):
            assert (data.x[n, 3] >= -pi)
            assert (data.x[n, 3] <= pi)

if __name__ == "__main__":
    demo()
