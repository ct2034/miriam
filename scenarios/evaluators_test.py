import random
import unittest

import numpy as np
import pytest

import scenarios.evaluators
from scenarios import test_helper

TEST_TIMEOUT = 5  # to be used for ecbs and icts calls


class TestEvaluators(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # making the folder to store data for test in.
        test_helper.make_cache_folder_and_set_envvar()

    @classmethod
    def tearDownClass(cls):
        # remove the folder that the test stored data in.
        test_helper.remove_cache_folder_and_unset_envvar()

    def test_to_agent_objects(self):
        # generating successfully
        starts = np.array([
            [0, 0],
            [0, 2]
        ])
        goals = np.array([
            [2, 0],
            [2, 2]
        ])
        res_agents = scenarios.evaluators.to_agent_objects(
            test_helper.env, starts, goals)
        res_starts = list(map(lambda a: tuple(a.pos), res_agents))
        res_goals = list(map(lambda a: tuple(a.goal), res_agents))
        self.assertIn((0, 0), res_starts)
        self.assertIn((0, 2), res_starts)
        self.assertIn((2, 0), res_goals)
        self.assertIn((2, 2), res_goals)
        # returns invalid when one path is not possible
        starts_invalid = np.array([
            [0, 0],
            [1, 0]  # obstacle
        ])
        goals_invalid = np.array([
            [0, 2],
            [1, 2]  # obstacle
        ])
        self.assertEqual(
            scenarios.evaluators.INVALID,
            scenarios.evaluators.to_agent_objects(
                test_helper.env, starts_invalid, goals_invalid)
        )

    def test_is_well_formed(self):
        # well formed
        starts_wf = np.array([
            [0, 0],
            [0, 2]
        ])
        goals_wf = np.array([
            [2, 0],
            [2, 2]
        ])
        self.assertTrue(
            scenarios.evaluators.is_well_formed(
                test_helper.env, starts_wf, goals_wf)
        )
        # not well formed
        starts_nwf = np.array([
            [1, 1],
            [0, 2]
        ])
        goals_nwf = np.array([
            [2, 0],
            [2, 2]
        ])
        self.assertFalse(
            scenarios.evaluators.is_well_formed(
                test_helper.env, starts_nwf, goals_nwf)
        )
        # returns invalid when one path is not possible
        starts_invalid = np.array([
            [0, 0],
            [1, 0]  # obstacle
        ])
        goals_invalid = np.array([
            [0, 2],
            [1, 2]  # obstacle
        ])
        self.assertFalse(
            scenarios.evaluators.is_well_formed(
                test_helper.env, starts_invalid, goals_invalid)
        )

    def test_cost_ecbs_success(self):
        # agents that collide in the middle
        starts_4_5 = np.array([
            [0, 0],
            [0, 2]
        ])
        goals_4_5 = np.array([
            [2, 0],
            [2, 2]
        ])
        self.assertAlmostEqual(
            4.5, scenarios.evaluators.cost_ecbs(
                test_helper.env, starts_4_5, goals_4_5)
        )
        # agents that don't collide
        self.assertAlmostEqual(
            2, scenarios.evaluators.cost_ecbs(
                test_helper.env, test_helper.starts_no_collision,
                test_helper.goals_no_collision)
        )

    def test_cost_ecbs_invalid(self):
        # one agents path is not possible
        starts_invalid = np.array([
            [0, 0],
            [1, 0]  # obstacle
        ])
        goals_invalid = np.array([
            [0, 2],
            [1, 2]  # obstacle
        ])
        self.assertAlmostEqual(
            scenarios.evaluators.INVALID,
            scenarios.evaluators.cost_ecbs(
                test_helper.env, starts_invalid, goals_invalid)
        )

    @pytest.mark.timeout(TEST_TIMEOUT)
    def test_cost_ecbs_timeout(self):
        # trying to make is time out ...
        starts_timeout = np.array([
            [0, 0],
            [0, 1],
            [0, 2],
            [1, 1],
            [2, 0],
            [2, 1],
            [2, 2]
        ])
        goals_timeout = np.array([
            [2, 0],
            [2, 1],
            [2, 2],
            [0, 0],
            [0, 1],
            [0, 2],
            [1, 1]
        ])
        self.assertEqual(
            scenarios.evaluators.INVALID,
            scenarios.evaluators.cost_ecbs(
                test_helper.env, starts_timeout, goals_timeout,
                timeout=TEST_TIMEOUT/2)
        )

    def test_blocks_ecbs(self):
        # agents that collide on vertex
        starts_vertex = np.array([
            [0, 0],
            [0, 2]
        ])
        goals_vertex = np.array([
            [2, 0],
            [2, 2]
        ])
        self.assertEqual(
            (1, 0), scenarios.evaluators.blocks_ecbs(
                test_helper.env, starts_vertex, goals_vertex)
        )

        # agents that collide on edge
        starts_edge = np.array([
            [0, 0],
            [2, 1]
        ])
        goals_edge = np.array([
            [2, 0],
            [0, 2]
        ])
        self.assertEqual(
            (1, 1), scenarios.evaluators.blocks_ecbs(
                test_helper.env, starts_edge, goals_edge)
        )

    def test_blocks_ecbs_invalid(self):
        # unsolvable
        starts_unsolv = np.array([
            [1, 0]  # obstacle
        ])
        goals_unsolv = np.array([
            [1, 2]  # obstacle
        ])
        self.assertEqual(
            scenarios.evaluators.INVALID, scenarios.evaluators.blocks_ecbs(
                test_helper.env, starts_unsolv, goals_unsolv)
        )

    def test_expanded_nodes_ecbs_no_collision(self):
        # agents that don't collide should not expand ecbs nodes
        self.assertEqual(
            1, scenarios.evaluators.expanded_nodes_ecbs(
                test_helper.env, test_helper.starts_no_collision,
                test_helper.goals_no_collision)
        )

    def test_expanded_nodes_ecbs_invalid(self):
        # unsolvable
        starts_unsolv = np.array([
            [1, 0]  # obstacle
        ])
        goals_unsolv = np.array([
            [1, 2]  # obstacle
        ])
        self.assertEqual(
            scenarios.evaluators.INVALID,
            scenarios.evaluators.expanded_nodes_ecbs(
                test_helper.env, starts_unsolv, goals_unsolv)
        )

    def test_cost_independent(self):
        # agents that would collide in the middle
        starts_4 = np.array([
            [0, 0],
            [0, 2]
        ])
        goals_4 = np.array([
            [2, 0],
            [2, 2]
        ])
        self.assertAlmostEqual(
            4, scenarios.evaluators.cost_independent(
                test_helper.env, starts_4, goals_4)
        )
        # agents that don't collide
        self.assertAlmostEqual(
            2, scenarios.evaluators.cost_independent(
                test_helper.env, test_helper.starts_no_collision,
                test_helper.goals_no_collision)
        )

    def test_cost_independent_invalid(self):
        # one agents path is not possible
        starts_invalid = np.array([
            [0, 0],
            [1, 0]  # obstacle
        ])
        goals_invalid = np.array([
            [0, 2],
            [2, 2]
        ])
        self.assertAlmostEqual(
            scenarios.evaluators.INVALID,
            scenarios.evaluators.cost_independent(
                test_helper.env, starts_invalid, goals_invalid)
        )

    def test_cost_sim_decentralized_random_collision(self):
        # agents that collide in the middle
        starts_4_5 = np.array([
            [0, 0],
            [0, 2]
        ])
        goals_4_5 = np.array([
            [2, 0],
            [2, 2]
        ])
        self.assertAlmostEqual(
            4.5, scenarios.evaluators.cost_sim_decentralized_random(
                test_helper.env, starts_4_5, goals_4_5)
        )

    def test_cost_sim_decentralized_random_no_collision(self):
        # agents that don't collide
        self.assertAlmostEqual(
            2, scenarios.evaluators.cost_sim_decentralized_random(
                test_helper.env, test_helper.starts_no_collision,
                test_helper.goals_no_collision)
        )

    def test_cost_sim_decentralized_random_unsolvable(self):
        starts_unsolv = np.array([
            [1, 0]  # obstacle
        ])
        goals_unsolv = np.array([
            [1, 2]  # obstacle
        ])
        self.assertEqual(
            scenarios.evaluators.INVALID,
            scenarios.evaluators.cost_sim_decentralized_random(
                test_helper.env, starts_unsolv, goals_unsolv)
        )

    def test_cost_sim_decentralized_random_deadlocks(self):
        # trying to make deadlocks ...
        random.seed(1)  # this some times times out on total random
        starts_deadlocks = np.array([
            [0, 0],
            [0, 1],
            [0, 2],
            [2, 2]
        ])
        goals_deadlocks = np.array([
            [2, 0],
            [2, 1],
            [2, 2],
            [0, 2]
        ])
        self.assertEqual(
            scenarios.evaluators.INVALID,
            scenarios.evaluators.cost_sim_decentralized_random(
                test_helper.env, starts_deadlocks, goals_deadlocks)
        )

    def test_expanded_nodes_icts_no_collision(self):
        # agents that don't collide should not expand icts nodes
        self.assertEqual(
            1, scenarios.evaluators.expanded_nodes_icts(
                test_helper.env, test_helper.starts_no_collision,
                test_helper.goals_no_collision)
        )

    def test_expanded_nodes_icts_deadlock(self):
        self.assertEqual(
            scenarios.evaluators.INVALID,
            scenarios.evaluators.expanded_nodes_icts(
                test_helper.env_deadlock, test_helper.starts_deadlock,
                test_helper.goals_deadlock, timeout=TEST_TIMEOUT)
        )

    def test_cost_icts_no_collision(self):
        self.assertAlmostEqual(
            2, scenarios.evaluators.cost_icts(
                test_helper.env, test_helper.starts_no_collision,
                test_helper.goals_no_collision)
        )

    def test_cost_icts_deadlock(self):
        self.assertEqual(
            scenarios.evaluators.INVALID,
            scenarios.evaluators.cost_icts(
                test_helper.env_deadlock, test_helper.starts_deadlock,
                test_helper.goals_deadlock, timeout=TEST_TIMEOUT)
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
