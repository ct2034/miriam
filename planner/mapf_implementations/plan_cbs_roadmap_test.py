import json
import os
import pickle
import unittest
from random import Random

import networkx as nx
import numpy as np
import pytest
import torch

from definitions import INVALID, POS
from planner.mapf_implementations.plan_cbs_roadmap import plan_cbsr
from planner.policylearn.edge_policy import EdgePolicyModel
from sim.decentralized.iterators import IteratorType
from sim.decentralized.policy import LearnedPolicy, PolicyType
from sim.decentralized.runner import run_a_scenario, to_agent_objects


class PlanCbsrTest(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName=methodName)
        self.g = nx.Graph()
        self.g.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])
        nx.set_node_attributes(
            self.g, {0: (0, 0), 1: (1, 0), 2: (1, 1), 3: (0, 1)}, POS
        )

    def test_solvable_scenario(self):
        """Simple solvable scenario."""
        starts = [0, 1]
        goals = [2, 3]
        paths = plan_cbsr(self.g, starts, goals, 0.2, 60, skip_cache=True)
        self.assertNotEqual(paths, INVALID)
        self.assertEqual(len(paths), 2)  # n_agents

    def test_one_agent_no_path(self):
        """Testing behavior when there is no path between start and goal."""
        g = self.g.copy()
        g.add_node(4)  # no connection to anything
        g.nodes[4][POS] = (2, 2)  # far away
        starts = [0]
        goals = [4]
        paths = plan_cbsr(g, starts, goals, 0.2, 60, skip_cache=True)
        # this currently times out which gives the right result but is not
        # really what we want
        self.assertEqual(paths, INVALID)

    def test_two_agents_same_start_or_goal(self):
        """Testing behavior when two agents start or end at the same place."""
        g = self.g.copy()
        # same start
        starts = [0, 0, 1, 2]
        goals = [2, 3, 0, 1]
        paths = plan_cbsr(
            g, starts, goals, 0.2, 1, skip_cache=True, ignore_finished_agents=False
        )
        self.assertEqual(paths, INVALID)
        # same goal
        starts = [0, 1, 2, 3]
        goals = [0, 1, 0, 2]
        paths = plan_cbsr(
            g, starts, goals, 0.2, 1, skip_cache=True, ignore_finished_agents=False
        )
        self.assertEqual(paths, INVALID)

    def test_scenario_solvable_with_waiting(self):
        """Scenario with waiting."""
        g = nx.Graph()
        g.add_edges_from([(0, 1), (1, 2), (2, 3), (2, 4), (4, 5)])
        nx.set_node_attributes(
            g,
            {
                0: (0, 0),
                1: (1, 0),
                2: (2, 0),
                3: (2, 1),  # waiting point
                4: (3, 0),
                5: (4, 0),
            },
            POS,
        )
        starts = [0, 5]
        goals = [5, 0]
        paths = plan_cbsr(g, starts, goals, 0.2, 60, skip_cache=True)
        self.assertNotEqual(paths, INVALID)
        self.assertEqual(len(paths), 2)  # n_agents
        waited = False
        for i_a in range(2):
            prev_node = None
            for node in paths[i_a]:
                if prev_node is not None:
                    if prev_node[0] == node[0]:
                        waited = True
                prev_node = node
        self.assertTrue(waited)

    def test_requires_ignore_finished_agent(self):
        """Specific scenarios should be solvable iff finished agents are ignored."""
        g = nx.Graph()
        g.add_edges_from([(0, 1), (1, 2), (2, 3)])
        nx.set_node_attributes(g, {0: (0, 0), 1: (1, 0), 2: (2, 0), 3: (3, 0)}, POS)
        starts = [0, 3]
        goals = [2, 1]

        # solvable
        paths = plan_cbsr(
            g, starts, goals, 0.2, 1, skip_cache=True, ignore_finished_agents=True
        )
        self.assertNotEqual(paths, INVALID)

        # not solvable
        paths = plan_cbsr(
            g, starts, goals, 0.2, 1, skip_cache=True, ignore_finished_agents=False
        )
        self.assertEqual(paths, INVALID)

    def tests_optimality(self):
        DIRECTORY = os.path.join(os.path.dirname(__file__), "test_cases")
        for file_or_dir in os.listdir(DIRECTORY):
            dir = os.path.join(DIRECTORY, file_or_dir)
            if os.path.isdir(dir) and file_or_dir.startswith("optimality"):
                print("Testing optimality of", file_or_dir)

                rng = Random(0)
                # graph = nx.read_gpickle(os.path.join(dir, f"{file_or_dir}.gpickle"))
                with open(os.path.join(dir, f"{file_or_dir}.gpickle"), "rb") as f:
                    graph = pickle.load(f)
                assert isinstance(graph, nx.Graph)
                model = EdgePolicyModel()
                model.load_state_dict(
                    torch.load(os.path.join(dir, f"{file_or_dir}.pt"))
                )
                starts_goals = None
                with open(os.path.join(dir, f"starts_goals.json")) as f:
                    starts_goals = json.load(f)
                assert starts_goals is not None
                starts = np.array(starts_goals["starts"])
                goals = np.array(starts_goals["goals"])
                radius = 0.001
                edges = np.array(list(graph.edges.data("distance")))
                non_self_edges = edges[edges[:, 0] != edges[:, 1]]
                min_edge_length = min(non_self_edges[:, 2])

                # run
                paths_cbsr_raw = plan_cbsr(
                    graph,
                    starts,
                    goals,
                    radius,
                    60,
                    skip_cache=True,
                    ignore_finished_agents=True,
                )
                assert isinstance(paths_cbsr_raw, list)
                max_len = max(map(len, paths_cbsr_raw))
                paths_cbsr = []
                for p in paths_cbsr_raw:
                    p_out = []
                    for x in p:
                        p_out.append(x[0])
                    p_out.extend(
                        [
                            p_out[-1],
                        ]
                        * (max_len - len(p_out))
                    )
                    paths_cbsr.append(p_out)
                print(f"{paths_cbsr=}")
                print("-" * 80)

                paths_sim_learned = []
                agents_sim_learned = to_agent_objects(
                    graph, starts.tolist(), goals.tolist(), radius=radius, rng=Random(0)
                )
                assert isinstance(agents_sim_learned, list)
                for a in agents_sim_learned:
                    a.policy = LearnedPolicy(a, model)
                res_sim_learned = run_a_scenario(
                    graph,
                    agents_sim_learned,
                    False,
                    iterator=IteratorType.LOOKAHEAD2,
                    paths_out=paths_sim_learned,
                )
                print(f"{res_sim_learned=}")
                print(f"{paths_sim_learned=}")
                print("-" * 80)

                paths_sim_optimal = []
                agents_sim_optimal = to_agent_objects(
                    graph,
                    starts.tolist(),
                    goals.tolist(),
                    policy=PolicyType.OPTIMAL,
                    radius=radius,
                    rng=rng,
                )
                assert isinstance(agents_sim_optimal, list)
                res_sim_optimal = run_a_scenario(
                    graph,
                    agents_sim_optimal,
                    False,
                    iterator=IteratorType.LOOKAHEAD2,
                    paths_out=paths_sim_optimal,
                )
                print(f"{res_sim_optimal=}")
                print(f"{paths_sim_optimal=}")
                print("-" * 80)

                # check optimality
                timed_len_optimal = (
                    res_sim_optimal[2] + res_sim_optimal[0] * min_edge_length
                )
                timed_len_learned = (
                    res_sim_learned[2] + res_sim_learned[0] * min_edge_length
                )
                self.assertGreaterEqual(timed_len_learned, timed_len_optimal)


if __name__ == "__main__":
    pytest.main(["-s", "-v", __file__])
