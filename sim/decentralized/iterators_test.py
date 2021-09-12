#!/usr/bin/env python3
import unittest

import pytest
from sim.decentralized.iterators import *
from sim.decentralized.policy import PolicyType


class TestIterators(unittest.TestCase):
    def test_check_for_colissions_and_get_possible_next_agent_poses(self):
        env = np.array([[0, 0], [0, 1]])
        agents = (
            Agent(env, np.array([0, 0]), PolicyType.RANDOM),
            Agent(env, np.array([0, 1]), PolicyType.RANDOM),
            Agent(env, np.array([1, 0]), PolicyType.RANDOM)
        )
        agents[0].give_a_goal(np.array([0, 1]))
        agents[1].give_a_goal(np.array([0, 0]))
        agents[2].give_a_goal(np.array([0, 0]))

        # getting next steps when all are allowed
        possible_next_agent_poses = get_possible_next_agent_poses(
            agents, [True] * 3
        )
        self.assertEqual(len(possible_next_agent_poses), 3)
        self.assertEqual(len(possible_next_agent_poses[0]), 2)
        self.assertTrue(possible_next_agent_poses[0] == (0, 1))
        self.assertTrue(possible_next_agent_poses[1] == (0, 0))
        self.assertTrue(possible_next_agent_poses[2] == (0, 0))

        # getting next steps when none are allowed
        other_possible_next_agent_poses = get_possible_next_agent_poses(
            agents, [False] * 3
        )
        self.assertTrue(
            all(other_possible_next_agent_poses[0] == np.array([0, 0])))
        self.assertTrue(
            all(other_possible_next_agent_poses[1] == np.array([0, 1])))
        self.assertTrue(
            all(other_possible_next_agent_poses[2] == np.array([1, 0])))

        # checking for collisions
        node_colissions, edge_colissions = check_for_colissions(
            agents)
        self.assertListEqual(list(node_colissions.keys()), [(0, 0)])
        self.assertIn(1, node_colissions[(0, 0)])
        self.assertIn(2, node_colissions[(0, 0)])
        edge_in_col = ((0, 0), (0, 1))
        self.assertListEqual(list(edge_colissions.keys()), [edge_in_col])
        self.assertIn(0, edge_colissions[edge_in_col])
        self.assertIn(1, edge_colissions[edge_in_col])


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
