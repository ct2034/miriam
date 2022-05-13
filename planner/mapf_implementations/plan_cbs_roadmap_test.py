import unittest

import networkx as nx
from definitions import INVALID, POS
from planner.mapf_implementations.plan_cbs_roadmap import plan_cbsr


class PlanCbsrTest(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName=methodName)
        self.g = nx.Graph()
        self.g.add_edges_from([
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (0, 2)
        ])
        nx.set_node_attributes(self.g, {
            0: (0, 0),
            1: (1, 0),
            2: (1, 1),
            3: (0, 1)
        }, POS)

    def test_solvable_scenario(self):
        """Simple solvable scenario."""
        starts = [0, 1]
        goals = [2, 3]
        paths = plan_cbsr(self.g, starts, goals, .2, 60, skip_cache=True)
        self.assertNotEqual(paths, INVALID)
        self.assertEqual(len(paths), 2)  # n_agents

    def test_one_agent_no_path(self):
        """Testing behavior when there is no path between start and goal."""
        g = self.g.copy()
        g.add_node(4)  # no connection to anything
        g.nodes[4][POS] = (2, 2)  # far away
        starts = [0]
        goals = [4]
        paths = plan_cbsr(g, starts, goals, .2, 60, skip_cache=True)
        # this currently times out which gives the right result but is not
        # really what we want
        self.assertEqual(paths, INVALID)

    def test_two_agents_same_start_or_goal(self):
        """Testing behavior when two agents start or end at the same place."""
        g = self.g.copy()
        # same start
        starts = [0, 0, 1, 2]
        goals = [2, 3, 0, 1]
        paths = plan_cbsr(g, starts, goals, .2, 1,
                          skip_cache=True, ignore_finished_agents=False)
        self.assertEqual(paths, INVALID)
        # same goal
        starts = [0, 1, 2, 3]
        goals = [0, 1, 0, 2]
        paths = plan_cbsr(g, starts, goals, .2, 1,
                          skip_cache=True, ignore_finished_agents=False)
        self.assertEqual(paths, INVALID)

    def test_scenario_solvable_with_waiting(self):
        """Scenario with waiting."""
        g = nx.Graph()
        g.add_edges_from([
            (0, 1),
            (1, 2),
            (2, 3),
            (2, 4),
            (4, 5)
        ])
        nx.set_node_attributes(g, {
            0: (0, 0),
            1: (1, 0),
            2: (2, 0),
            3: (2, 1),  # waiting point
            4: (3, 0),
            5: (4, 0)
        }, POS)
        starts = [0, 5]
        goals = [5, 0]
        paths = plan_cbsr(g, starts, goals, .2, 60, skip_cache=True)
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
        g.add_edges_from([
            (0, 1),
            (1, 2),
            (2, 3)
        ])
        nx.set_node_attributes(g, {
            0: (0, 0),
            1: (1, 0),
            2: (2, 0),
            3: (3, 0)
        }, POS)
        starts = [0, 3]
        goals = [2, 1]

        # solvable
        paths = plan_cbsr(g, starts, goals, .2, 1,
                          skip_cache=True, ignore_finished_agents=True)
        self.assertNotEqual(paths, INVALID)

        # not solvable
        paths = plan_cbsr(g, starts, goals, .2, 1,
                          skip_cache=True, ignore_finished_agents=False)
        self.assertEqual(paths, INVALID)
