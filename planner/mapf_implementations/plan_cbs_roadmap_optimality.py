from random import Random

import networkx as nx
import numpy as np
import torch
from definitions import INVALID, POS
from learn.delaunay_benchmark.delaunay_implementations import (
    read_map, run_delaunay_libpysal)
from planner.mapf_implementations.plan_cbs_roadmap import plan_cbsr
from planner.policylearn.edge_policy import EdgePolicyModel
from sim.decentralized.iterators import IteratorType
from sim.decentralized.policy import LearnedPolicy
from sim.decentralized.runner import run_a_scenario, to_agent_objects

if __name__ == "__main__":
    # inputs
    rng = Random(0)
    data_no = 573
    graph = nx.read_gpickle(
        f"planner/mapf_implementations/{data_no}.gpickle")
    model = EdgePolicyModel()
    model.load_state_dict(
        torch.load(f"planner/mapf_implementations/{data_no}.pt"))
    starts = np.array([3, 1, 7])
    goals = np.array([0, 3, 5])
    radius = 0.001

    # run
    paths_cbsr = plan_cbsr(graph, starts, goals, radius, 60, skip_cache=True)
    assert(isinstance(paths_cbsr, list))
    print(paths_cbsr)

    paths_sim = []
    agents = to_agent_objects(
        graph, starts.tolist(), goals.tolist(),
        radius=radius, rng=rng)
    assert isinstance(agents, list)
    for a in agents:
        a.policy = LearnedPolicy(a, model)
    res_sim = run_a_scenario(
        graph, agents, False, iterator=IteratorType.LOOKAHEAD2, paths_out=paths_sim)
    print(paths_sim)
