from random import Random

import networkx as nx
import numpy as np
import torch
from definitions import IDX_SUCCESS
from matplotlib import pyplot as plt
from planner.mapf_implementations.plan_cbs_roadmap import plan_cbsr
from planner.policylearn.edge_policy import EdgePolicyModel
from scenarios.visualization import plot_with_paths
from sim.decentralized.iterators import IteratorType
from sim.decentralized.policy import LearnedPolicy, PolicyType
from sim.decentralized.runner import run_a_scenario, to_agent_objects

if __name__ == "__main__":
    # inputs
    rng = Random(0)
    data_no = "optimality"
    graph = nx.read_gpickle(
        f"planner/mapf_implementations/test_cases/{data_no}/{data_no}.gpickle")
    model = EdgePolicyModel()
    model.load_state_dict(
        torch.load(f"planner/mapf_implementations/test_cases/{data_no}/{data_no}.pt"))
    starts = np.array([13, 1, 6])
    goals = np.array([14, 10, 5])
    radius = 0.001

    # run
    paths_cbsr_raw = plan_cbsr(
        graph, starts, goals, radius, 60, skip_cache=True, ignore_finished_agents=True)
    assert(isinstance(paths_cbsr_raw, list))
    max_len = max(map(len, paths_cbsr_raw))
    paths_cbsr = []
    for p in paths_cbsr_raw:
        p_out = []
        for x in p:
            p_out.append(x[0])
        p_out.extend([p_out[-1], ] * (max_len - len(p_out)))
        paths_cbsr.append(p_out)
    print(f"{paths_cbsr=}")

    paths_sim_learned = []
    agents_sim_learned = to_agent_objects(
        graph, starts.tolist(), goals.tolist(),
        radius=radius, rng=rng)
    assert isinstance(agents_sim_learned, list)
    for a in agents_sim_learned:
        a.policy = LearnedPolicy(a, model)
    res_sim_learned = run_a_scenario(
        graph, agents_sim_learned, False, iterator=IteratorType.LOOKAHEAD2, paths_out=paths_sim_learned)
    print(f"{res_sim_learned=}")
    print(f"{paths_sim_learned=}")

    paths_sim_optimal = []
    agents_sim_optimal = to_agent_objects(
        graph, starts.tolist(), goals.tolist(),
        policy=PolicyType.OPTIMAL,
        radius=radius, rng=rng)
    assert isinstance(agents_sim_optimal, list)
    res_sim_optimal = run_a_scenario(
        graph, agents_sim_optimal, False, iterator=IteratorType.LOOKAHEAD2, paths_out=paths_sim_optimal)
    print(f"{res_sim_optimal=}")
    print(f"{paths_sim_optimal=}")

    # plot
    fig_cbsr = plt.figure()
    fig_cbsr.suptitle('Paths CBSR')
    plot_with_paths(graph, paths_cbsr, fig=fig_cbsr)

    fig_learned = plt.figure()
    fig_learned.suptitle('Paths Learned Policy')
    plot_with_paths(graph, paths_sim_learned, fig=fig_learned)

    plt.show()
