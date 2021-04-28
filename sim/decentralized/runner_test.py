#!/usr/bin/env python3
import random
import unittest
from unittest.mock import MagicMock

import numpy as np
import pytest
import sim.decentralized.runner as runner
from sim.decentralized.agent import Agent
from sim.decentralized.iterators import IteratorType, get_iterator_fun
from sim.decentralized.policy import PolicyType, RandomPolicy
from sim.decentralized.runner import SimIterationException


class TestRunner(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestRunner, self).__init__(*args, **kwargs)
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

    def test_iterate_sim_and_are_all_agents_at_their_goals(self):
        for iterator_type in IteratorType:
            iterator_fun = get_iterator_fun(iterator_type)
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
            iterator_fun(agents)
            self.assertTrue(all(agents[0].pos == np.array([0, 1])))
            self.assertTrue(all(agents[1].pos == np.array([1, 0])))
            self.assertTrue(runner.are_all_agents_at_their_goals(agents))

            # after another iteration they should be still at their goal, raising
            # an exception for a node deadlock.
            self.assertRaises(SimIterationException,
                              lambda: iterator_fun(agents))
            self.assertTrue(all(agents[0].pos == np.array([0, 1])))
            self.assertTrue(all(agents[1].pos == np.array([1, 0])))
            self.assertTrue(runner.are_all_agents_at_their_goals(agents))

    def test_iterate_sim_with_node_coll(self):
        # as it is now, this only works with waiting
        waiting_iterator_fun = get_iterator_fun(IteratorType.WAITING)
        for prio0, prio1 in [(.7, .3), (.3, .7)]:
            env = np.array([[0, 0], [0, 1]])
            agents = (
                Agent(env, np.array([0, 1]), PolicyType.RANDOM),
                Agent(env, np.array([1, 0]), PolicyType.RANDOM)
            )
            agents[0].give_a_goal(np.array([0, 0]))
            agents[1].give_a_goal(np.array([0, 0]))

            agents[0].get_priority = MagicMock(return_value=prio0)
            agents[1].get_priority = MagicMock(return_value=prio1)

            waiting_iterator_fun(agents)
            if prio0 > prio1:
                # both want to go to `(0,0)` only the first (`0`) should do
                self.assertTrue(all(agents[0].pos == np.array([0, 0])))  # g
                self.assertTrue(all(agents[1].pos == np.array([1, 0])))  # s
            else:
                # both want to go to `(0,0)` only the second (`1`) should do
                self.assertTrue(all(agents[0].pos == np.array([0, 1])))  # s
                self.assertTrue(all(agents[1].pos == np.array([0, 0])))  # g

            # after another iteration both should be there and finished
            waiting_iterator_fun(agents)
            self.assertTrue(all(agents[0].pos == np.array([0, 0])))  # goal
            self.assertTrue(all(agents[1].pos == np.array([0, 0])))  # goal
            self.assertTrue(runner.are_all_agents_at_their_goals(agents))

    def test_iterate_sim_with_node_coll_deadlock(self):
        for iterator_type in IteratorType:
            iterator_fun = get_iterator_fun(iterator_type)
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
                              lambda: iterator_fun(agents))

    def test_iterate_sim_with_edge_coll(self):
        for iterator_type in IteratorType:
            iterator_fun = get_iterator_fun(iterator_type)
            env = np.array([[0, 0, 0, 0], [1, 1, 1, 1],
                            [1, 1, 1, 1], [1, 1, 1, 1]])
            agents = (
                Agent(env, np.array([0, 0]), PolicyType.RANDOM),
                Agent(env, np.array([0, 3]), PolicyType.RANDOM)
            )
            agents[0].give_a_goal(np.array([0, 3]))
            agents[1].give_a_goal(np.array([0, 0]))
            iterator_fun(agents)
            for _ in range(10):
                self.assertRaises(runner.SimIterationException,
                                  lambda: iterator_fun(agents))

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
