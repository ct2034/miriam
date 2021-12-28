#!/usr/bin/env python3
import random
import unittest
from random import Random
from unittest.mock import MagicMock

import networkx as nx
import numpy as np
import pytest
import sim.decentralized.runner as runner
from definitions import POS
from scenarios.generators import corridor_with_passing
from sim.decentralized.agent import Agent
from sim.decentralized.iterators import IteratorType, get_iterator_fun
from sim.decentralized.policy import PolicyType, RandomPolicy
from sim.decentralized.runner import (SimIterationException, run_a_scenario,
                                      to_agent_objects)
from tools import hasher


class TestRunner(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestRunner, self).__init__(*args, **kwargs)
        random.seed(0)

    def test_initialize_environment(self):
        env = runner.initialize_environment_random_fill(10, .5)
        self.assertEqual(env.shape, (10, 10))
        self.assertEqual(np.count_nonzero(env), 50)

    def test_initialize_environment_determinism(self):
        rng = random.Random(0)
        env = runner.initialize_environment_random_fill(10, .5, rng)
        rng = random.Random(0)
        env_same_seed = runner.initialize_environment_random_fill(10, .5, rng)
        rng = random.Random(1)
        env_different_seed = runner.initialize_environment_random_fill(
            10, .5, rng)
        self.assertEqual(hasher([env]), hasher([env_same_seed]))
        self.assertNotEqual(hasher([env]), hasher([env_different_seed]))

    def test_initialize_new_agent(self):
        env_zz = np.array([[0, 0], [1, 1]])
        zero_zero = runner.initialize_new_agent(
            env_zz, [], PolicyType.RANDOM, False)
        self.assertTrue((zero_zero.pos == (0, 0)) or
                        (zero_zero.goal == (0, 0)))
        self.assertTrue((zero_zero.pos == (0, 1)) or
                        (zero_zero.goal == (0, 1)))
        self.assertIsInstance(zero_zero.policy, RandomPolicy)

        env_zo = np.array([[1, 0], [1, 1]])
        zero_one = runner.initialize_new_agent(env_zo, [],
                                               PolicyType.RANDOM, True)
        self.assertTrue((zero_one.pos == (0, 1)))
        self.assertTrue((zero_one.goal == (0, 1)))
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

    def test_initialize_agents_determinism(self):
        env = runner.initialize_environment_random_fill(10, .5)
        agents = runner.initialize_agents(
            env, 5, PolicyType.LEARNED, True, random.Random(0))
        agents_same_seed = runner.initialize_agents(
            env, 5, PolicyType.LEARNED, True, random.Random(0))
        agents_different_seed = runner.initialize_agents(
            env, 5, PolicyType.LEARNED, True, random.Random(1))
        self.assertEqual(hasher(agents), hasher(agents_same_seed))
        self.assertNotEqual(hasher(agents), hasher(agents_different_seed))

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
        for iterator_type in [
            IteratorType.WAITING,
            IteratorType.BLOCKING1,
            IteratorType.BLOCKING3
        ]:
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
            iterator_fun(agents, True)
            self.assertTrue(all(agents[0].pos == np.array([0, 1])))
            self.assertTrue(all(agents[1].pos == np.array([1, 0])))
            self.assertTrue(runner.are_all_agents_at_their_goals(agents))

            # after another iteration they should be still at their goal, raising
            # an exception for a node deadlock.
            self.assertRaises(SimIterationException,
                              lambda: iterator_fun(agents, True))
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

            waiting_iterator_fun(agents, True)
            if prio0 > prio1:
                # both want to go to `(0,0)` only the first (`0`) should do
                self.assertTrue(all(agents[0].pos == np.array([0, 0])))  # g
                self.assertTrue(all(agents[1].pos == np.array([1, 0])))  # s
            else:
                # both want to go to `(0,0)` only the second (`1`) should do
                self.assertTrue(all(agents[0].pos == np.array([0, 1])))  # s
                self.assertTrue(all(agents[1].pos == np.array([0, 0])))  # g

            # after another iteration both should be there and finished
            waiting_iterator_fun(agents, True)
            self.assertTrue(all(agents[0].pos == np.array([0, 0])))  # goal
            self.assertTrue(all(agents[1].pos == np.array([0, 0])))  # goal
            self.assertTrue(runner.are_all_agents_at_their_goals(agents))

    @pytest.mark.skip
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
                              lambda: iterator_fun(agents, True))

    @pytest.mark.skip
    def test_iterate_sim_with_edge_coll(self):
        for iterator_type in IteratorType:
            iterator_fun = get_iterator_fun(iterator_type)
            env = np.ones((8, 8), dtype=int)
            env[0, :] = 0
            agents = (
                Agent(env, np.array([0, 0]), PolicyType.RANDOM),
                Agent(env, np.array([0, 7]), PolicyType.RANDOM)
            )
            agents[0].give_a_goal(np.array([0, 7]))
            agents[1].give_a_goal(np.array([0, 0]))
            iterator_fun(agents, True)
            # BLOCKING3 will see the edge collision after first iteration,
            # the others not.
            if iterator_type != IteratorType.BLOCKING3:
                iterator_fun(agents)
                iterator_fun(agents)
            for _ in range(10):
                self.assertRaises(runner.SimIterationException,
                                  lambda: iterator_fun(agents, True))

    def test_run_a_scenario_oscillation_detection(self):
        i_r = 3
        rng = random.Random(i_r)
        (env, starts, goals) = corridor_with_passing(
            10, 0, 2, rng)
        agents = to_agent_objects(
            env, starts, goals, PolicyType.RANDOM, rng)
        from sim.decentralized.iterators import SimIterationException
        SimIterationException.__init__ = MagicMock(return_value=None)
        res = run_a_scenario(
            env, agents, False, IteratorType.BLOCKING1, ignore_finished_agents=False)
        SimIterationException.__init__.assert_called_with(
            "oscillation deadlock")
        print(res)

    def test_evaluate_policies(self):
        n_runs = 5
        for agents in [2, 3]:
            size = 4
            data, names = runner.evaluate_policies(size, agents, n_runs, False)
            # assert len(data) == len(PolicyType)  # not testing all policies
            for p in data.keys():
                dat = data[p]
                (n_evals_out, n_runs_out) = dat.shape
                assert n_runs_out == n_runs
                assert n_evals_out == len(names)
                for i_e in range(n_evals_out):
                    if names[i_e] != "successful":
                        # data should be different
                        assert np.std(dat[i_e, :]) > 0

    def test_running_on_graph(self):
        """Making sure iterator works with agents on graph"""
        rng = Random(0)
        pos_s = {
            0: (0., 0.),
            1: (0., 1.),
            2: (1., 0.),
            3: (1., 1.),
            4: (.5, .5)
        }
        g = nx.Graph()
        g.add_edge(0, 1)
        g.add_edge(0, 2)
        g.add_edge(1, 3)
        g.add_edge(2, 3)
        g.add_edge(3, 4)
        g.add_edge(4, 0)
        g.add_edge(4, 1)
        g.add_edge(4, 2)
        nx.set_node_attributes(g, pos_s, POS)

        agents = [
            Agent(g, 0, PolicyType.RANDOM, rng=rng),
            Agent(g, 1, PolicyType.RANDOM, rng=rng)
        ]
        agents[0].give_a_goal(3)
        agents[1].give_a_goal(2)

        paths_out = []
        res = run_a_scenario(
            None, agents, False, IteratorType.BLOCKING1,
            paths_out=paths_out)
        pass


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
