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
        paths = plan_cbsr(self.g, starts, goals, .2, 60)
        self.assertNotEqual(paths, INVALID)
        self.assertEqual(len(paths), 2)  # n_agents

    def test_one_agent_no_path(self):
        """Testing behavior when there is no path between start and goal."""
        g = self.g.copy()
        g.add_node(4)  # no connection to anything
        g.nodes[4][POS] = (2, 2)  # far away
        starts = [0]
        goals = [4]
        paths = plan_cbsr(g, starts, goals, .2, 60)
        # this currently times out which gives the right result but is not
        # really what we want
        self.assertEqual(paths, INVALID)
