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
from scenarios.test_helper import make_cache_folder_and_set_envvar
from sim.decentralized.agent import Agent
from sim.decentralized.iterators import IteratorType, get_iterator_fun
from sim.decentralized.policy import PolicyType, RandomPolicy
from sim.decentralized.runner import (
    SimIterationException,
    run_a_scenario,
    to_agent_objects,
)
from tools import hasher


class TestRunner(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestRunner, self).__init__(*args, **kwargs)
        make_cache_folder_and_set_envvar()
        random.seed(0)

    def test_initialize_environment(self):
        env = runner.initialize_environment_random_fill(10, 0.5)
        self.assertEqual(env.shape, (10, 10))
        self.assertEqual(np.count_nonzero(env), 50)

    def test_initialize_environment_determinism(self):
        rng = random.Random(0)
        env = runner.initialize_environment_random_fill(10, 0.5, rng)
        rng = random.Random(0)
        env_same_seed = runner.initialize_environment_random_fill(10, 0.5, rng)
        rng = random.Random(1)
        env_different_seed = runner.initialize_environment_random_fill(10, 0.5, rng)
        self.assertEqual(hasher([env]), hasher([env_same_seed]))
        self.assertNotEqual(hasher([env]), hasher([env_different_seed]))

    def test_initialize_new_agent(self):
        env_zz = np.array([[0, 0], [1, 1]])
        zero_zero = runner.initialize_new_agent(env_zz, [], PolicyType.RANDOM, False)
        assert zero_zero is not None
        assert zero_zero.pos is not None
        assert zero_zero.goal is not None
        self.assertTrue((zero_zero.pos == 0) or (zero_zero.goal == 0))
        self.assertTrue((zero_zero.pos == 1) or (zero_zero.goal == 1))
        self.assertIsInstance(zero_zero.policy, RandomPolicy)

        env_zo = np.array([[1, 0], [1, 1]])
        zero_one = runner.initialize_new_agent(env_zo, [], PolicyType.RANDOM, True)
        assert zero_one is not None
        self.assertTrue((zero_one.pos == 1))
        self.assertTrue((zero_one.goal == 1))
        self.assertIsInstance(zero_one.policy, RandomPolicy)

    def test_initialize_agents(self):
        env = np.array([[0, 0], [0, 0]])
        agents = runner.initialize_agents(env, 2, PolicyType.RANDOM)
        assert agents is not None
        self.assertEqual(len(agents), 2)
        self.assertTrue(
            0 in map(lambda a: a.pos, agents) or 0 in map(lambda a: a.goal, agents)
        )
        self.assertTrue(
            1 in map(lambda a: a.pos, agents) or 1 in map(lambda a: a.goal, agents)
        )
        self.assertTrue(
            2 in map(lambda a: a.pos, agents) or 2 in map(lambda a: a.goal, agents)
        )
        self.assertTrue(
            3 in map(lambda a: a.pos, agents) or 3 in map(lambda a: a.goal, agents)
        )

    def test_initialize_agents_determinism(self):
        env = runner.initialize_environment_random_fill(10, 0.5)
        agents = runner.initialize_agents(
            env, 5, PolicyType.LEARNED, True, random.Random(0)
        )
        agents_same_seed = runner.initialize_agents(
            env, 5, PolicyType.LEARNED, True, random.Random(0)
        )
        agents_different_seed = runner.initialize_agents(
            env, 5, PolicyType.LEARNED, True, random.Random(1)
        )
        self.assertEqual(hasher(agents), hasher(agents_same_seed))
        self.assertNotEqual(hasher(agents), hasher(agents_different_seed))

    def test_initialize_agents_tight_placement(self):
        env = np.array([[0, 0], [1, 1]])
        agents = runner.initialize_agents(
            env, 2, PolicyType.RANDOM, tight_placement=True
        )
        assert agents is not None
        self.assertEqual(len(agents), 2)
        self.assertTrue(
            0 in map(lambda a: a.pos, agents) and 0 in map(lambda a: a.goal, agents)
        )
        self.assertTrue(
            1 in map(lambda a: a.pos, agents) and 1 in map(lambda a: a.goal, agents)
        )

    def test_iterate_sim_and_are_all_agents_at_their_goals(self):
        for iterator_type in IteratorType:
            iterator_fun = get_iterator_fun(iterator_type)
            env = np.array([[0, 0], [0, 0]])
            agents = (
                Agent(env, np.array([0, 0]), PolicyType.RANDOM),
                Agent(env, np.array([1, 1]), PolicyType.RANDOM),
            )
            agents[0].give_a_goal(np.array([0, 1]))
            agents[1].give_a_goal(np.array([1, 0]))

            # first we should not be at the goal
            self.assertFalse(runner.are_all_agents_at_their_goals(agents))

            # after one iteration all agents should be at their goal
            iterator_fun(agents, True)
            self.assertTrue(agents[0].pos == 1)
            self.assertTrue(agents[1].pos == 2)
            self.assertTrue(runner.are_all_agents_at_their_goals(agents))

            # after another iteration they should be still at their goal, raising
            # an exception for a node deadlock.
            # TODO: currently no exception is raised here, but it should be, maybe
            # self.assertRaises(SimIterationException,
            #                   lambda: iterator_fun(agents, True))
            self.assertTrue(agents[0].pos == 1)
            self.assertTrue(agents[1].pos == 2)
            self.assertTrue(runner.are_all_agents_at_their_goals(agents))

    @pytest.mark.skip
    def test_iterate_sim_with_node_coll(self):
        # as it is now, this only works with waiting
        waiting_iterator_fun = get_iterator_fun(IteratorType.LOOKAHEAD1)
        for prio0, prio1 in [(0.7, 0.3), (0.3, 0.7)]:
            env = np.array([[0, 0], [0, 1]])
            agents = (
                Agent(env, np.array([0, 1]), PolicyType.RANDOM),
                Agent(env, np.array([1, 0]), PolicyType.RANDOM),
            )
            agents[0].give_a_goal(np.array([0, 0]))
            agents[1].give_a_goal(np.array([0, 0]))

            # TODO: no priorities any more
            agents[0].get_priority = MagicMock(return_value=prio0)
            agents[1].get_priority = MagicMock(return_value=prio1)

            waiting_iterator_fun(agents, True)
            if prio0 > prio1:
                # both want to go to `(0,0)` only the first (`0`) should do
                self.assertTrue(agents[0].pos == 0)  # g
                self.assertTrue(agents[1].pos == 2)  # s
            else:
                # both want to go to `(0,0)` only the second (`1`) should do
                self.assertTrue(agents[0].pos == 1)  # s
                self.assertTrue(agents[1].pos == 0)  # g

            # after another iteration both should be there and finished
            waiting_iterator_fun(agents, True)
            self.assertTrue(agents[0].pos == 0)  # goal
            self.assertTrue(agents[1].pos == 0)  # goal
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
                Agent(env, np.array([2, 0]), PolicyType.RANDOM),
            )
            agents[0].give_a_goal(np.array([0, 1]))
            agents[1].give_a_goal(np.array([1, 1]))
            agents[2].give_a_goal(np.array([1, 0]))
            agents[3].give_a_goal(np.array([0, 0]))
            agents[4].give_a_goal(np.array([1, 0]))
            agents[4].get_priority = MagicMock(return_value=1)

            self.assertRaises(
                runner.SimIterationException, lambda: iterator_fun(agents, True)
            )

    @pytest.mark.skip
    def test_iterate_sim_with_edge_coll(self):
        for iterator_type in IteratorType:
            iterator_fun = get_iterator_fun(iterator_type)
            env = np.ones((8, 8), dtype=int)
            env[0, :] = 0
            agents = (
                Agent(env, np.array([0, 0]), PolicyType.RANDOM),
                Agent(env, np.array([0, 7]), PolicyType.RANDOM),
            )
            agents[0].give_a_goal(np.array([0, 7]))
            agents[1].give_a_goal(np.array([0, 0]))
            iterator_fun(agents, True)
            # LOOKAHEAD3 will see the edge collision after first iteration,
            # the others not.
            if iterator_type != IteratorType.LOOKAHEAD3:
                iterator_fun(agents)
                iterator_fun(agents)
            for _ in range(10):
                self.assertRaises(
                    runner.SimIterationException, lambda: iterator_fun(agents, True)
                )

    @pytest.mark.skip(reason="Not sure how to fix this, TODO")
    def test_run_a_scenario_oscillation_detection(self):
        i_r = 0
        rng = random.Random(i_r)
        (env, starts, goals) = corridor_with_passing(10, 0, 2, rng)
        agents = to_agent_objects(env, starts, goals, policy=PolicyType.RANDOM, rng=rng)
        from sim.decentralized.iterators import SimIterationException

        SimIterationException.__init__ = MagicMock(return_value=None)
        res = run_a_scenario(
            env, agents, False, IteratorType.LOOKAHEAD1, ignore_finished_agents=False
        )
        SimIterationException.__init__.assert_called_with("oscillation deadlock")
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
            0: (0.0, 0.0),
            1: (0.0, 1.0),
            2: (1.0, 0.0),
            3: (1.0, 1.0),
            4: (0.5, 0.5),
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
            Agent(g, 0, PolicyType.RANDOM, rng=rng, radius=0.2),
            Agent(g, 1, PolicyType.RANDOM, rng=rng, radius=0.2),
        ]
        agents[0].give_a_goal(3)
        agents[1].give_a_goal(2)

        paths_out = []
        _ = run_a_scenario(
            None, agents, False, IteratorType.LOOKAHEAD1, paths_out=paths_out
        )

        # check paths_out
        self.assertEqual(len(paths_out), 2)
        self.assertEqual(len(paths_out[0]), 4)
        self.assertEqual(len(paths_out[1]), 4)


if __name__ == "__main__":  # pragma: no cover
    pytest.main(["-vs", __file__])
