#!/usr/bin/env python3

import random
import unittest
from unittest.mock import MagicMock

import numpy as np
import pytest
import sim.decentralized.runner as runner
from sim.decentralized.agent import Agent
from sim.decentralized.policy import PolicyType, RandomPolicy
from sim.decentralized.runner import SimIterationException


class TestDecentralizedSim(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestDecentralizedSim, self).__init__(*args, **kwargs)
        random.seed(0)

    def test_initialize_environment(self):
        env = runner.initialize_environment(10, .5)
        self.assertEqual(env.shape, (10, 10))
        self.assertEqual(np.count_nonzero(env), 50)

    def test_initialize_new_agent(self):
        env_zz = np.array([[0, 0], [1, 1]])
        zero_zero = runner.initialize_new_agent(
            env_zz, [], PolicyType.RANDOM, False)
        self.assertTrue((zero_zero.pos == [0, 0]).all() or
                        (zero_zero.goal == [0, 0]).all())
        self.assertTrue((zero_zero.pos == [0, 1]).all() or
                        (zero_zero.goal == [0, 1]).all())
        self.assertIsInstance(zero_zero.policy, RandomPolicy)

        env_zo = np.array([[1, 0], [1, 1]])
        zero_one = runner.initialize_new_agent(env_zo, [],
                                               PolicyType.RANDOM, True)
        self.assertTrue((zero_one.pos == [0, 1]).all())
        self.assertTrue((zero_one.goal == [0, 1]).all())
        self.assertIsInstance(zero_one.policy, RandomPolicy)

    def test_initialize_agents(self):
        env = np.array([[0, 0], [0, 0]])
        agents = runner.initialize_agents(env, 2, PolicyType.RANDOM)
        self.assertEqual(len(agents), 2)
        self.assertTrue((0, 0) in map(lambda a: tuple(a.pos), agents) or
                        (0, 0) in map(lambda a: tuple(a.goal), agents))
        self.assertTrue((0, 1) in map(lambda a: tuple(a.pos), agents) or
                        (0, 1) in map(lambda a: tuple(a.goal), agents))
        self.assertTrue((1, 0) in map(lambda a: tuple(a.pos), agents) or
                        (1, 0) in map(lambda a: tuple(a.goal), agents))
        self.assertTrue((1, 1) in map(lambda a: tuple(a.pos), agents) or
                        (1, 1) in map(lambda a: tuple(a.goal), agents))

    def test_initialize_agents_tight_placement(self):
        env = np.array([[0, 0], [1, 1]])
        agents = runner.initialize_agents(
            env, 2, PolicyType.RANDOM, tight_placement=True)
        self.assertEqual(len(agents), 2)
        self.assertTrue((0, 0) in map(lambda a: tuple(a.pos), agents) and
                        (0, 0) in map(lambda a: tuple(a.goal), agents))
        self.assertTrue((0, 1) in map(lambda a: tuple(a.pos), agents) and
                        (0, 1) in map(lambda a: tuple(a.goal), agents))

    def test_is_environment_well_formed(self):
        env = np.array([[0, 0, 0, 0], [1, 0, 1, 0],
                        [1, 0, 1, 0], [1, 1, 1, 1]])

        # both agents going down left and right
        agents = (
            Agent(env, np.array([0, 1])),
            Agent(env, np.array([0, 3]))
        )
        agents[0].give_a_goal(np.array([2, 1]))
        agents[1].give_a_goal(np.array([2, 3]))
        self.assertTrue(runner.is_environment_well_formed(agents))

        # one goal in the middle
        agents = (
            Agent(env, np.array([2, 1])),
            Agent(env, np.array([0, 0]))
        )
        agents[0].give_a_goal(np.array([0, 2]))  # top middle
        agents[1].give_a_goal(np.array([0, 3]))
        self.assertFalse(runner.is_environment_well_formed(agents))

        # one start in the middle
        agents = (
            Agent(env, np.array([0, 2])),  # top middle
            Agent(env, np.array([0, 0]))
        )
        agents[0].give_a_goal(np.array([2, 3]))
        agents[1].give_a_goal(np.array([0, 3]))
        self.assertFalse(runner.is_environment_well_formed(agents))

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
        possible_next_agent_poses = runner.get_possible_next_agent_poses(
            agents, [True] * 3
        )
        self.assertEqual(possible_next_agent_poses.shape, (3, 2))
        self.assertTrue(all(possible_next_agent_poses[0] == np.array([0, 1])))
        self.assertTrue(all(possible_next_agent_poses[1] == np.array([0, 0])))
        self.assertTrue(all(possible_next_agent_poses[2] == np.array([0, 0])))

        # getting next steps when none are allowed
        other_possible_next_agent_poses = runner.get_possible_next_agent_poses(
            agents, [False] * 3
        )
        self.assertTrue(
            all(other_possible_next_agent_poses[0] == np.array([0, 0])))
        self.assertTrue(
            all(other_possible_next_agent_poses[1] == np.array([0, 1])))
        self.assertTrue(
            all(other_possible_next_agent_poses[2] == np.array([1, 0])))

        # checking for collisions
        node_colissions, edge_colissions = runner.check_for_colissions(
            agents, possible_next_agent_poses)
        self.assertListEqual(list(node_colissions.keys()), [(0, 0)])
        self.assertIn(1, node_colissions[(0, 0)])
        self.assertIn(2, node_colissions[(0, 0)])
        edge_in_col = ((0, 0), (0, 1))
        self.assertListEqual(list(edge_colissions.keys()), [edge_in_col])
        self.assertIn(0, edge_colissions[edge_in_col])
        self.assertIn(1, edge_colissions[edge_in_col])

    def test_iterate_sim_and_are_all_agents_at_their_goals(self):
        env = np.array([[0, 0], [0, 0]])
        agents = (
            Agent(env, np.array([0, 0]), PolicyType.RANDOM),
            Agent(env, np.array([1, 1]), PolicyType.RANDOM)
        )
        agents[0].give_a_goal(np.array([0, 1]))
        agents[1].give_a_goal(np.array([1, 0]))

        # first we should not be at the goal
        self.assertFalse(runner.are_all_agents_at_their_goals(agents))

        # after one iteration all agents should be at their goal
        runner.iterate_sim(agents)
        self.assertTrue(all(agents[0].pos == np.array([0, 1])))
        self.assertTrue(all(agents[1].pos == np.array([1, 0])))
        self.assertTrue(runner.are_all_agents_at_their_goals(agents))

        # after another iteration they should be still at their goal, raising
        # an exception for a node deadlock.
        self.assertRaises(SimIterationException,
                          lambda: runner.iterate_sim(agents))
        self.assertTrue(all(agents[0].pos == np.array([0, 1])))
        self.assertTrue(all(agents[1].pos == np.array([1, 0])))
        self.assertTrue(runner.are_all_agents_at_their_goals(agents))

    def test_iterate_sim_with_node_coll(self):
        env = np.array([[0, 0], [0, 1]])
        agents = (
            Agent(env, np.array([0, 1]), PolicyType.RANDOM),
            Agent(env, np.array([1, 0]), PolicyType.RANDOM)
        )
        agents[0].give_a_goal(np.array([0, 0]))
        agents[1].give_a_goal(np.array([0, 0]))

        agents[0].get_priority = MagicMock(return_value=.7)
        agents[1].get_priority = MagicMock(return_value=.3)

        # both agents want to go to `(0,0)` only the first should get there
        runner.iterate_sim(agents)
        self.assertTrue(all(agents[0].pos == np.array([0, 0])))  # goal
        self.assertTrue(all(agents[1].pos == np.array([1, 0])))  # start

        # after another iteration both should be there and finished
        runner.iterate_sim(agents)
        self.assertTrue(all(agents[0].pos == np.array([0, 0])))  # goal
        self.assertTrue(all(agents[1].pos == np.array([0, 0])))  # goal
        self.assertTrue(runner.are_all_agents_at_their_goals(agents))

    def test_iterate_sim_with_node_coll_reverse(self):
        # new agents for reverse prios
        env = np.array([[0, 0], [0, 1]])
        agents = (
            Agent(env, np.array([0, 1]), PolicyType.RANDOM),
            Agent(env, np.array([1, 0]), PolicyType.RANDOM)
        )
        agents[0].give_a_goal(np.array([0, 0]))
        agents[1].give_a_goal(np.array([0, 0]))

        # inverse priorities
        agents[0].get_priority = MagicMock(return_value=0)
        agents[1].get_priority = MagicMock(return_value=.9)

        # both agents want to go to `(0,0)` only the second should get there
        runner.iterate_sim(agents)
        self.assertTrue(all(agents[0].pos == np.array([0, 1])))  # start
        self.assertTrue(all(agents[1].pos == np.array([0, 0])))  # goal

        # after another iteration both should be there and finished
        runner.iterate_sim(agents)
        self.assertTrue(all(agents[0].pos == np.array([0, 0])))  # goal
        self.assertTrue(all(agents[1].pos == np.array([0, 0])))  # goal
        self.assertTrue(runner.are_all_agents_at_their_goals(agents))

    def test_iterate_sim_with_node_coll_deadlock(self):
        env = np.array([[0, 0, 1], [0, 0, 1], [0, 1, 1]])
        agents = (
            Agent(env, np.array([0, 0]), PolicyType.RANDOM),
            Agent(env, np.array([0, 1]), PolicyType.RANDOM),
            Agent(env, np.array([1, 1]), PolicyType.RANDOM),
            Agent(env, np.array([1, 0]), PolicyType.RANDOM),
            Agent(env, np.array([2, 0]), PolicyType.RANDOM)
        )
        agents[0].give_a_goal(np.array([0, 1]))
        agents[1].give_a_goal(np.array([1, 1]))
        agents[2].give_a_goal(np.array([1, 0]))
        agents[3].give_a_goal(np.array([0, 0]))
        agents[4].give_a_goal(np.array([1, 0]))
        agents[4].get_priority = MagicMock(return_value=1)

        self.assertRaises(runner.SimIterationException,
                          lambda: runner.iterate_sim(agents))

    def test_iterate_sim_with_edge_coll(self):
        env = np.array([[0, 0, 0, 0], [1, 1, 1, 1],
                        [1, 1, 1, 1], [1, 1, 1, 1]])
        agents = (
            Agent(env, np.array([0, 0]), PolicyType.RANDOM),
            Agent(env, np.array([0, 3]), PolicyType.RANDOM)
        )
        agents[0].give_a_goal(np.array([0, 3]))
        agents[1].give_a_goal(np.array([0, 0]))
        runner.iterate_sim(agents)
        for _ in range(10):
            self.assertRaises(runner.SimIterationException,
                              lambda: runner.iterate_sim(agents))

    def test_evaluate_policies(self):
        n_runs = 4
        for agents in [2, 3, 4]:
            data, names = runner.evaluate_policies(10, agents, n_runs, False)
            assert len(data) == len(PolicyType)
            for p in data.keys():
                dat = data[p]
                (n_evals_out, n_runs_out) = dat.shape
                assert n_runs_out == n_runs
                assert n_evals_out == len(names)
                for i_e in range(n_evals_out):
                    if names[i_e] != "successful":
                        assert np.std(dat[i_e, :]) > 0


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
