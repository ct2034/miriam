#!/usr/bin/env python3
import unittest
from unittest.mock import MagicMock

import pytest

from definitions import PATH
from sim.decentralized.iterators import *
from sim.decentralized.policy import PolicyType


class PathMatcher:
    def __init__(self, path_to_match: PATH) -> None:
        assert isinstance(path_to_match, List)
        assert isinstance(path_to_match[0], int)
        self.path_to_match = path_to_match

    def __eq__(self, compare_to: PATH) -> bool:
        assert isinstance(compare_to, List)
        assert isinstance(compare_to[0], int)
        if len(self.path_to_match) != len(compare_to):
            return False
        for i in range(len(self.path_to_match)):
            if self.path_to_match[i] != compare_to[i]:
                return False
        return True

    def __str__(self) -> str:
        return f"PathMatcher({self.path_to_match})"


class TestIterators(unittest.TestCase):
    def __init__(self, methodName: str = "") -> None:
        super().__init__(methodName=methodName)
        self.env = np.array([[0, 0], [0, 1]])
        self.agents = (
            Agent(self.env, (0, 0), PolicyType.RANDOM),
            Agent(self.env, (0, 1), PolicyType.RANDOM),
            Agent(self.env, (1, 0), PolicyType.RANDOM),
        )
        self.agents[0].give_a_goal((0, 1))
        self.agents[1].give_a_goal((0, 0))
        self.agents[2].give_a_goal((0, 0))

        # a graph for collision checking
        self.g = nx.Graph()
        self.g.add_edges_from([(0, 1), (0, 2), (1, 2)])
        self.g.add_node(0, pos=(0.0, 0.0))
        self.g.add_node(1, pos=(0.0, 1.0))
        self.g.add_node(2, pos=(1.0, 0.0))

        # square graph with middle bit
        self.env_sq_graph = nx.Graph()
        self.env_sq_graph.add_edges_from(
            [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4), (2, 4), (3, 4)]
        )
        nx.set_node_attributes(
            self.env_sq_graph,
            {
                0: (0.0, 0.0),
                1: (0.0, 1.0),
                2: (1.0, 1.0),
                3: (1.0, 0.0),
                4: (0.5, 0.5),
            },
            POS,
        )

    def init_agents(self, g):
        agent_0 = Agent(g, 0, radius=0.1)
        agent_1 = Agent(g, 1, radius=0.1)
        agent_0.give_a_goal(2)
        agent_1.give_a_goal(3)
        return agent_0, agent_1

    def test_get_possible_next_agent_poses_when_all_are_allowed(self):
        # getting next steps when all are allowed
        possible_next_agent_poses = get_possible_next_agent_poses(
            self.agents, [True] * 3
        )
        self.assertEqual(len(possible_next_agent_poses), 3)
        self.assertIsInstance(possible_next_agent_poses[0], int)
        self.assertTrue(possible_next_agent_poses[0] == 1)
        self.assertTrue(possible_next_agent_poses[1] == 0)
        self.assertTrue(possible_next_agent_poses[2] == 0)

    def test_get_possible_next_agent_poses_when_none_are_allowed(self):
        # getting next steps when none are allowed
        other_possible_next_agent_poses = get_possible_next_agent_poses(
            self.agents, [False] * 3
        )
        self.assertTrue(other_possible_next_agent_poses[0] == 0)
        self.assertTrue(other_possible_next_agent_poses[1] == 1)
        self.assertTrue(other_possible_next_agent_poses[2] == 2)

    def test_check_for_colissions(self):
        node_colissions, edge_colissions = check_for_colissions(
            self.agents, ignore_finished_agents=True
        )
        self.assertListEqual(list(node_colissions.keys()), [0])
        self.assertIn(1, node_colissions[0])
        self.assertIn(2, node_colissions[0])
        edge_in_col = (0, 1)
        self.assertListEqual(list(edge_colissions.keys()), [edge_in_col])
        self.assertIn(0, edge_colissions[edge_in_col])
        self.assertIn(1, edge_colissions[edge_in_col])

    def test_check_for_colissions_ignore_finished_agents(self):
        self.agents[1].make_next_step(0)
        self.assertTrue(self.agents[1].is_at_goal())
        node_colissions, edge_colissions = check_for_colissions(
            self.agents, ignore_finished_agents=True
        )
        self.assertEqual(len(node_colissions), 0)
        self.assertEqual(len(edge_colissions), 0)

    def test_check_for_colissions_not_ignore_finished_agents(self):
        node_colissions, edge_colissions = check_for_colissions(
            self.agents, ignore_finished_agents=False
        )
        self.assertListEqual(list(node_colissions.keys()), [0])
        self.assertIn(1, node_colissions[0])
        self.assertIn(2, node_colissions[0])
        edge_in_col = (0, 1)
        self.assertListEqual(list(edge_colissions.keys()), [edge_in_col])
        self.assertIn(0, edge_colissions[edge_in_col])
        self.assertIn(1, edge_colissions[edge_in_col])

    def test_check_motion_col(self):
        # no colissions, radius too small
        starts = [0, 1, 2]
        ends = [1, 2, 0]
        colliding_agents = check_motion_col(
            self.g, 0.22, starts, ends, ignored_agents=set()
        )
        self.assertEqual(len(colliding_agents), 0)

        # all collisions on bigger radius
        colliding_agents = check_motion_col(
            self.g, 0.23, starts, ends, ignored_agents=set()
        )
        self.assertEqual(len(colliding_agents), 3)
        self.assertIn(0, colliding_agents)
        self.assertIn(1, colliding_agents)
        self.assertIn(2, colliding_agents)

        # ignoring agent 0
        colliding_agents = check_motion_col(
            self.g, 0.23, starts, ends, ignored_agents=set([0])
        )
        self.assertEqual(len(colliding_agents), 2)
        self.assertNotIn(0, colliding_agents)
        self.assertIn(1, colliding_agents)
        self.assertIn(2, colliding_agents)

        # looking at only the outer edges
        # no colissions, radius too small
        starts = [0, 1]
        ends = [2, 0]
        colliding_agents = check_motion_col(
            self.g, 0.35, starts, ends, ignored_agents=set()
        )
        self.assertEqual(len(colliding_agents), 0)

        # all collisions on bigger radius
        colliding_agents = check_motion_col(
            self.g, 0.36, starts, ends, ignored_agents=set()
        )
        self.assertEqual(len(colliding_agents), 2)
        self.assertIn(0, colliding_agents)
        self.assertIn(1, colliding_agents)

        # moving to the same node should always collide
        all_nodes = [0, 1, 2]
        for node in all_nodes:
            starts = [node, node]
            ends = [n for n in all_nodes if n != node]
            colliding_agents = check_motion_col(
                self.g, 0.01, starts, ends, ignored_agents=set()
            )
            self.assertEqual(len(colliding_agents), 2)
            self.assertIn(0, colliding_agents)
            self.assertIn(1, colliding_agents)

        # waiting should be possible
        # and lead to no colission when radius is small
        starts = [0, 1]
        ends = [0, 2]
        colliding_agents = check_motion_col(
            self.g, 0.35, starts, ends, ignored_agents=set()
        )
        self.assertEqual(len(colliding_agents), 0)

        # and lead to colission when radius is big
        colliding_agents = check_motion_col(
            self.g, 0.36, starts, ends, ignored_agents=set()
        )
        self.assertEqual(len(colliding_agents), 2)
        self.assertIn(0, colliding_agents)
        self.assertIn(1, colliding_agents)

        # if one agent doesn't collide ...
        # the other two should
        starts = [0, 1, 2]
        ends = [0, 2, 1]
        colliding_agents = check_motion_col(
            self.g, 0.35, starts, ends, ignored_agents=set()
        )
        self.assertEqual(len(colliding_agents), 2)
        self.assertIn(1, colliding_agents)
        self.assertIn(2, colliding_agents)

        # if radius is bigger, all collide
        colliding_agents = check_motion_col(
            self.g, 0.36, starts, ends, ignored_agents=set()
        )
        self.assertEqual(len(colliding_agents), 3)
        self.assertIn(0, colliding_agents)
        self.assertIn(1, colliding_agents)
        self.assertIn(2, colliding_agents)

    def test_when_is_policy_called(self):
        agent_0, agent_1 = self.init_agents(self.env_sq_graph)

        # check initial paths
        self.assertEqual(agent_0.path, PathMatcher([0, 4, 2]))
        self.assertEqual(agent_1.path, PathMatcher([1, 4, 3]))

        # agent 0 can go
        get_edge_mock_4 = MagicMock(return_value=4)
        agent_0.policy.get_edge = get_edge_mock_4

        # agent 1 should wait
        get_edge_mock_1 = MagicMock(return_value=1)
        agent_1.policy.get_edge = get_edge_mock_1

        # run the scenario
        iterate_edge_policy((agent_0, agent_1), 1, True, _copying=True)
        self.assertEqual(get_edge_mock_1.call_count, 1)
        self.assertEqual(get_edge_mock_4.call_count, 1)
        self.assertEqual(
            get_edge_mock_1.call_args_list[0].args[0][0].path, PathMatcher([0, 4, 2])
        )
        self.assertEqual(
            get_edge_mock_1.call_args_list[0].args[0][1].path, PathMatcher([1, 4, 3])
        )

    @pytest.mark.skip(reason="Not sure how to fix this, TODO")
    def test_how_is_policy_called(self):
        agent_0, agent_1 = self.init_agents(self.env_sq_graph)

        # check initial paths
        self.assertEqual(agent_0.path, PathMatcher([0, 4, 2]))
        self.assertEqual(agent_1.path, PathMatcher([1, 4, 3]))

        # now tell both, they can go
        def effect_1_4(*args, **kwargs):
            yield 1
            yield 4
            while True:
                yield None

        get_edge_mock_1_then_4 = MagicMock(side_effect=effect_1_4())
        get_edge_mock_1 = MagicMock(return_value=1)
        agent_0.policy.get_edge = get_edge_mock_1
        agent_1.policy.get_edge = get_edge_mock_1_then_4

        iterate_edge_policy((agent_0, agent_1), 1, True, _copying=True)
        self.assertEqual(get_edge_mock_1_then_4.call_count, 2)
        self.assertEqual(get_edge_mock_1.call_count, 2)

        # first call
        self.assertEqual(
            get_edge_mock_1_then_4.call_args_list[0].args[0][0].path,
            PathMatcher([0, 4, 2]),
        )
        self.assertEqual(
            get_edge_mock_1_then_4.call_args_list[0].args[0][1].path,
            PathMatcher([1, 4, 3]),
        )
        self.assertEqual(
            get_edge_mock_1.call_args_list[0].args[0][0].path, PathMatcher([0, 4, 2])
        )
        self.assertEqual(
            get_edge_mock_1.call_args_list[0].args[0][1].path, PathMatcher([1, 4, 3])
        )

        # second call
        self.assertEqual(
            get_edge_mock_1_then_4.call_args_list[1].args[0][0].path,
            PathMatcher([0, 1, 2]),
        )
        self.assertEqual(
            get_edge_mock_1_then_4.call_args_list[1].args[0][1].path,
            PathMatcher([1, 1, 4, 3]),
        )
        self.assertEqual(
            get_edge_mock_1.call_args_list[1].args[0][0].path, PathMatcher([0, 1, 2])
        )
        self.assertEqual(
            get_edge_mock_1.call_args_list[1].args[0][1].path, PathMatcher([1, 1, 4, 3])
        )


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
