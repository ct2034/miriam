from itertools import product
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import networkx as nx
import torch
from definitions import IDX_AVERAGE_LENGTH, IDX_SUCCESS, PATH
from multi_optim.dagger import RADIUS
from planner.policylearn.edge_policy import EdgePolicyModel
from scenarios.visualization import plot_with_paths
from sim.decentralized.agent import Agent
from sim.decentralized.policy import LearnedPolicy, PolicyType
from sim.decentralized.runner import SCENARIO_RESULT, run_a_scenario
from tools import ProgressBar

if __name__ == '__main__':
    n_agents = 8
    n_nodes = 64

    n_runs = 1

    # policies
    policies = [
        PolicyType.LEARNED,
        PolicyType.RANDOM,
        PolicyType.OPTIMAL]

    results = {}  # type: Dict[int, Dict[PolicyType, SCENARIO_RESULT]]
    paths = {}  # type: Dict[int, Dict[PolicyType, List[PATH]]]

    g = nx.read_gpickle("debug.gpickle")
    policy_model = EdgePolicyModel()
    policy_model.load_state_dict(torch.load("debug.pt"))

    pb = ProgressBar("", n_runs * len(policies))
    for i_r, policy in product(range(n_runs), policies):
        if i_r not in results.keys():
            results[i_r] = {}
        if i_r not in paths.keys():
            paths[i_r] = {}

        # into a scenario
        starts = [2, 4, 0]  # type: ignore
        goals = [4, 0, 3]  # type: ignore
        # plot_env_with_arrows(g, starts, goals)  # type: ignore

        # initialize agents
        agents = tuple([Agent(g, start, radius=RADIUS, policy=policy)
                       for start in starts])
        for i, agent in enumerate(agents):
            agent.give_a_goal(goals[i])
            if policy == PolicyType.LEARNED:
                agent.policy = LearnedPolicy(agent, policy_model)

        # independent paths
        # do_collide, paths = will_agents_collide(
        #     agents, ignore_finished_agents=True)
        # print(f"Will agents collide? {do_collide}")
        # plot_with_paths(g, paths)

        # run the scenario
        paths_run: List[Any] = []
        res = run_a_scenario(env=g,
                             agents=agents,
                             plot=False,
                             paths_out=paths_run)
        paths[i_r][policy] = paths_run
        results[i_r][policy] = res

        if (PolicyType.LEARNED in results[i_r].keys() and
                PolicyType.OPTIMAL in results[i_r].keys()):
            if (results[i_r][PolicyType.LEARNED][IDX_AVERAGE_LENGTH] <
                    results[i_r][PolicyType.OPTIMAL][IDX_AVERAGE_LENGTH] and
                    results[i_r][PolicyType.LEARNED][IDX_SUCCESS] and
                    results[i_r][PolicyType.OPTIMAL][IDX_SUCCESS]):
                print(f"Policy {policy} is better than optimal")
                # filename of roadmap is 'planner/mapf_implementations/cache/187280093900cae76fffd919cbea9110bf8042fab2df6aa35bbd415480b785ce_roadmap.yaml'
                # paths_learned = [[2, **1**, 2, 4], [4, 2, 6, 0], [0, 5, 3, 3]]
                # paths_optimal = [[2, **3**, 2, 4], [4, 2, 6, 0], [0, 5, 3, 3]]
                # which are both possible, if i analyzed the roadmap files correctly. But the learned is shorter ...
                print(repr(starts))
                print(repr(goals))
                plot_with_paths(g, paths[i_r][PolicyType.LEARNED])
                plot_with_paths(g, paths[i_r][PolicyType.OPTIMAL])
                plt.show()
        pb.progress()

    pb.end()
