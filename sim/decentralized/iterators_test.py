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

    def test_get_possible_next_agent_poses_when_all_are_allowed(self):
        # getting next steps when all are allowed
        possible_next_agent_poses = get_possible_next_agent_poses(
            self.agents, [True] * 3
        )
        self.assertEqual(len(possible_next_agent_poses), 3)
        self.assertEqual(len(possible_next_agent_poses[0]), 2)
        self.assertTrue(possible_next_agent_poses[0] == (0, 1))
        self.assertTrue(possible_next_agent_poses[1] == (0, 0))
        self.assertTrue(possible_next_agent_poses[2] == (0, 0))

    def test_get_possible_next_agent_poses_when_none_are_allowed(self):
        # getting next steps when none are allowed
        other_possible_next_agent_poses = get_possible_next_agent_poses(
            self.agents, [False] * 3
        )
        self.assertTrue(
            all(other_possible_next_agent_poses[0] == np.array([0, 0])))
        self.assertTrue(
            all(other_possible_next_agent_poses[1] == np.array([0, 1])))
        self.assertTrue(
            all(other_possible_next_agent_poses[2] == np.array([1, 0])))

    def test_check_for_colissions(self):
        node_colissions, edge_colissions = check_for_colissions(
            self.agents, ignore_finished_agents=True)
        self.assertListEqual(list(node_colissions.keys()), [(0, 0)])
        self.assertIn(1, node_colissions[(0, 0)])
        self.assertIn(2, node_colissions[(0, 0)])
        edge_in_col = ((0, 0), (0, 1))
        self.assertListEqual(list(edge_colissions.keys()), [edge_in_col])
        self.assertIn(0, edge_colissions[edge_in_col])
        self.assertIn(1, edge_colissions[edge_in_col])

    def test_check_for_colissions_ignore_finished_agents(self):
        self.agents[1].make_next_step((0, 0))
        self.assertTrue(self.agents[1].is_at_goal())
        node_colissions, edge_colissions = check_for_colissions(
            self.agents, ignore_finished_agents=True)
        self.assertEqual(len(node_colissions), 0)
        self.assertEqual(len(edge_colissions), 0)

    def test_check_for_colissions_not_ignore_finished_agents(self):
        node_colissions, edge_colissions = check_for_colissions(
            self.agents, ignore_finished_agents=False)
        self.assertListEqual(list(node_colissions.keys()), [(0, 0)])
        self.assertIn(1, node_colissions[(0, 0)])
        self.assertIn(2, node_colissions[(0, 0)])
        edge_in_col = ((0, 0), (0, 1))
        self.assertListEqual(list(edge_colissions.keys()), [edge_in_col])
        self.assertIn(0, edge_colissions[edge_in_col])
        self.assertIn(1, edge_colissions[edge_in_col])


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
