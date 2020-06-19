#!/usr/bin/env python3

import random
import unittest
from unittest.mock import MagicMock

import numpy as np

import agent
import sim
from agent import Agent, Policy


class TestDecentralizedSim(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestDecentralizedSim, self).__init__(*args, **kwargs)
        random.seed(0)

    def test_initialize_environment(self):
        env = sim.initialize_environment(10, .5)
        self.assertEqual(env.shape, (10, 10))
        self.assertEqual(np.count_nonzero(env), 50)

    def test_initialize_new_agent(self):
        env_zz = np.array([[0, 1], [1, 1]])
        zero_zero = sim.initialize_new_agent(env_zz, [], Policy.RANDOM)
        self.assertTrue((zero_zero.pos == [0, 0]).all())
        self.assertTrue((zero_zero.goal == [0, 0]).all())
        self.assertTrue(zero_zero.policy == Policy.RANDOM)
        env_zo = np.array([[0, 0], [1, 1]])
        zero_one = sim.initialize_new_agent(env_zo, [zero_zero],
                                            Policy.RANDOM)
        self.assertTrue((zero_one.pos == [0, 1]).all())
        self.assertTrue((zero_one.goal == [0, 1]).all())
        self.assertTrue(zero_one.policy == Policy.RANDOM)

    def test_initialize_agents(self):
        env = np.array([[0, 0], [0, 1]])
        agents = sim.initialize_agents(env, 3, Policy.RANDOM)
        self.assertEqual(len(agents), 3)
        self.assertIn((0, 0), map(lambda a: tuple(a.pos), agents))
        self.assertIn((0, 1), map(lambda a: tuple(a.pos), agents))
        self.assertIn((1, 0), map(lambda a: tuple(a.pos), agents))
        self.assertIn((0, 0), map(lambda a: tuple(a.goal), agents))
        self.assertIn((0, 1), map(lambda a: tuple(a.goal), agents))
        self.assertIn((1, 0), map(lambda a: tuple(a.goal), agents))

    def test_check_for_colissions_and_get_possible_next_agent_poses(self):
        env = np.array([[0, 0], [0, 1]])
        agents = [
            Agent(env, [0, 0], Policy.RANDOM),
            Agent(env, [0, 1], Policy.RANDOM),
            Agent(env, [1, 0], Policy.RANDOM)
        ]
        agents[0].give_a_goal(np.array([0, 1]))
        agents[1].give_a_goal(np.array([0, 0]))
        agents[2].give_a_goal(np.array([0, 0]))

        # getting next steps when all are allowed
        possible_next_agent_poses = sim.get_possible_next_agent_poses(
            agents, [True] * 3
        )
        self.assertTrue(all(possible_next_agent_poses[0] == np.array([0, 1])))
        self.assertTrue(all(possible_next_agent_poses[1] == np.array([0, 0])))
        self.assertTrue(all(possible_next_agent_poses[2] == np.array([0, 0])))

        # getting next steps when none are allowed
        other_possible_next_agent_poses = sim.get_possible_next_agent_poses(
            agents, [False] * 3
        )
        self.assertTrue(
            all(other_possible_next_agent_poses[0] == np.array([0, 0])))
        self.assertTrue(
            all(other_possible_next_agent_poses[1] == np.array([0, 1])))
        self.assertTrue(
            all(other_possible_next_agent_poses[2] == np.array([1, 0])))

        # checking for collisions
        node_colissions, edge_colissions = sim.check_for_colissions(
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
        agents = [
            Agent(env, [0, 0], Policy.RANDOM),
            Agent(env, [1, 1], Policy.RANDOM)
        ]
        agents[0].give_a_goal(np.array([0, 1]))
        agents[1].give_a_goal(np.array([1, 0]))

        # first we should not be at the goal
        self.assertFalse(sim.are_all_agents_at_their_goals(agents))

        # after one iteration all agents should be at their goal
        sim.iterate_sim(agents)
        self.assertTrue(all(agents[0].pos == np.array([0, 1])))
        self.assertTrue(all(agents[1].pos == np.array([1, 0])))
        self.assertTrue(sim.are_all_agents_at_their_goals(agents))

        # after another iteration they should be still at their goal
        sim.iterate_sim(agents)
        self.assertTrue(all(agents[0].pos == np.array([0, 1])))
        self.assertTrue(all(agents[1].pos == np.array([1, 0])))
        self.assertTrue(sim.are_all_agents_at_their_goals(agents))

    def test_iterate_sim_with_node_coll(self):
        env = np.array([[0, 0], [0, 1]])
        agents = [
            Agent(env, [0, 1], Policy.RANDOM),
            Agent(env, [1, 0], Policy.RANDOM)
        ]
        agents[0].give_a_goal(np.array([0, 0]))
        agents[1].give_a_goal(np.array([0, 0]))

        agents[0].get_priority = MagicMock(return_value=.7)
        agents[1].get_priority = MagicMock(return_value=.3)

        # both agents want to go to `(0,0)` only the first should get there
        sim.iterate_sim(agents)
        self.assertTrue(all(agents[0].pos == np.array([0, 0])))  # goal
        self.assertTrue(all(agents[1].pos == np.array([1, 0])))  # start

        # after another iteration both should be there and finished
        sim.iterate_sim(agents)
        self.assertTrue(all(agents[0].pos == np.array([0, 0])))  # goal
        self.assertTrue(all(agents[1].pos == np.array([0, 0])))  # goal
        self.assertTrue(sim.are_all_agents_at_their_goals(agents))

        # new agents for reverse prios
        agents = [
            Agent(env, [0, 1], Policy.RANDOM),
            Agent(env, [1, 0], Policy.RANDOM)
        ]
        agents[0].give_a_goal(np.array([0, 0]))
        agents[1].give_a_goal(np.array([0, 0]))

        # inverse priorities
        agents[0].get_priority = MagicMock(return_value=0)
        agents[1].get_priority = MagicMock(return_value=.9)

        # both agents want to go to `(0,0)` only the second should get there
        sim.iterate_sim(agents)
        self.assertTrue(all(agents[0].pos == np.array([0, 1])))  # start
        self.assertTrue(all(agents[1].pos == np.array([0, 0])))  # goal

        # after another iteration both should be there and finished
        sim.iterate_sim(agents)
        self.assertTrue(all(agents[0].pos == np.array([0, 0])))  # goal
        self.assertTrue(all(agents[1].pos == np.array([0, 0])))  # goal
        self.assertTrue(sim.are_all_agents_at_their_goals(agents))

    def test_iterate_sim_with_edge_coll(self):
        env = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
        agents = [
            Agent(env, [0, 0], Policy.RANDOM),
            Agent(env, [0, 3], Policy.RANDOM)
        ]
        agents[0].give_a_goal(np.array([0, 3]))
        agents[1].give_a_goal(np.array([0, 0]))
        sim.iterate_sim(agents)
        for _ in range(100):
            self.assertRaises(sim.SimIterationException, lambda: sim.iterate_sim(agents))

    def test_run_main(self):
        sim.run_main(5, 50, False, False)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
