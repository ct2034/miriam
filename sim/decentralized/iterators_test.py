#!/usr/bin/env python3
import unittest

import pytest
from sim.decentralized.iterators import *
from sim.decentralized.policy import PolicyType


class TestIterators(unittest.TestCase):
    def __init__(self, methodName: str = "") -> None:
        super().__init__(methodName=methodName)
        self.env = np.array([[0, 0], [0, 1]])
        self.agents = (
            Agent(self.env, (0, 0), PolicyType.RANDOM),
            Agent(self.env, (0, 1), PolicyType.RANDOM),
            Agent(self.env, (1, 0), PolicyType.RANDOM)
        )
        self.agents[0].give_a_goal((0, 1))
        self.agents[1].give_a_goal((0, 0))
        self.agents[2].give_a_goal((0, 0))

        # a graph for collision checking
        self.g = nx.Graph()
        self.g.add_edges_from([(0, 1), (0, 2), (1, 2)])
        self.g.add_node(0, pos=(0., 0.))
        self.g.add_node(1, pos=(0., 1.))
        self.g.add_node(2, pos=(1., 0.))

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
            self.agents, ignore_finished_agents=True)
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
            self.agents, ignore_finished_agents=True)
        self.assertEqual(len(node_colissions), 0)
        self.assertEqual(len(edge_colissions), 0)

    def test_check_for_colissions_not_ignore_finished_agents(self):
        node_colissions, edge_colissions = check_for_colissions(
            self.agents, ignore_finished_agents=False)
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
            self.g, 0.22, starts, ends, ignored_agents=set())
        self.assertEqual(len(colliding_agents), 0)

        # all collisions on bigger radius
        colliding_agents = check_motion_col(
            self.g, 0.23, starts, ends, ignored_agents=set())
        self.assertEqual(len(colliding_agents), 3)
        self.assertIn(0, colliding_agents)
        self.assertIn(1, colliding_agents)
        self.assertIn(2, colliding_agents)

        # looking at only the outer edges
        # no colissions, radius too small
        starts = [0, 1]
        ends = [2, 0]
        colliding_agents = check_motion_col(
            self.g, 0.35, starts, ends, ignored_agents=set())
        self.assertEqual(len(colliding_agents), 0)

        # all collisions on bigger radius
        colliding_agents = check_motion_col(
            self.g, 0.36, starts, ends, ignored_agents=set())
        self.assertEqual(len(colliding_agents), 2)
        self.assertIn(0, colliding_agents)
        self.assertIn(1, colliding_agents)

        # moving to the same node should always collide
        all_nodes = [0, 1, 2]
        for node in all_nodes:
            starts = [node, node]
            ends = [n for n in all_nodes if n != node]
            colliding_agents = check_motion_col(
                self.g, 0.01, starts, ends, ignored_agents=set())
            self.assertEqual(len(colliding_agents), 2)
            self.assertIn(0, colliding_agents)
            self.assertIn(1, colliding_agents)

        # waiting should be possible
        # and lead to no colission when radius is small
        starts = [0, 1]
        ends = [0, 2]
        colliding_agents = check_motion_col(
            self.g, 0.35, starts, ends, ignored_agents=set())
        self.assertEqual(len(colliding_agents), 0)

        # and lead to colission when radius is big
        colliding_agents = check_motion_col(
            self.g, 0.36, starts, ends, ignored_agents=set())
        self.assertEqual(len(colliding_agents), 2)
        self.assertIn(0, colliding_agents)
        self.assertIn(1, colliding_agents)

        # if one agent doesn't collide ...
        # the other two should
        starts = [0, 1, 2]
        ends = [0, 2, 1]
        colliding_agents = check_motion_col(
            self.g, 0.35, starts, ends, ignored_agents=set())
        self.assertEqual(len(colliding_agents), 2)
        self.assertIn(1, colliding_agents)
        self.assertIn(2, colliding_agents)

        # if radius is bigger, all collide
        colliding_agents = check_motion_col(
            self.g, 0.36, starts, ends, ignored_agents=set())
        self.assertEqual(len(colliding_agents), 3)
        self.assertIn(0, colliding_agents)
        self.assertIn(1, colliding_agents)
        self.assertIn(2, colliding_agents)


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
