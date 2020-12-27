import random
import unittest

import numpy as np

import scenarios.evaluators


class TestEvaluators(unittest.TestCase):
    env = np.array([
        [0, 0, 0],
        [1, 0, 1],
        [0, 0, 0]
    ])

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
            self.env, starts, goals, ignore_cache=True)
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
                self.env, starts_invalid, goals_invalid, ignore_cache=True)
        )

    def test_is_well_fromed(self):
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
                self.env, starts_wf, goals_wf, ignore_cache=True)
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
                self.env, starts_nwf, goals_nwf, ignore_cache=True)
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
                self.env, starts_invalid, goals_invalid, ignore_cache=True)
        )

    def test_plan_ecbs(self):
        # agents that collide in the middle
        starts_success = np.array([
            [0, 0],
            [0, 2]
        ])
        goals_success = np.array([
            [2, 0],
            [2, 2]
        ])
        res = scenarios.evaluators.plan_ecbs(
            self.env, starts_success, goals_success, ignore_cache=True)
        self.assertTrue(len(res.keys()) != 0)
        # one agents path is not possible
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
            scenarios.evaluators.plan_ecbs(
                self.env, starts_invalid, goals_invalid, ignore_cache=True)
        )
        # returns invalid when one agents path is not possible
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
            scenarios.evaluators.plan_ecbs(
                self.env, starts_invalid, goals_invalid, ignore_cache=True)
        )
        # trying to make deadlocks ...
        starts_deadlocks = np.array([
            [0, 0],
            [0, 1],
            [0, 2],
            [1, 1],
            [2, 0],
            [2, 1],
            [2, 2]
        ])
        goals_deadlocks = np.array([
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
            scenarios.evaluators.plan_ecbs(
                self.env, starts_deadlocks, goals_deadlocks, ignore_cache=True)
        )

    def test_cost_ecbs(self):
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
                self.env, starts_4_5, goals_4_5, ignore_cache=True)
        )
        # agents that don't collide
        starts_2 = np.array([
            [0, 0],
            [2, 0]
        ])
        goals_2 = np.array([
            [0, 2],
            [2, 2]
        ])
        self.assertAlmostEqual(
            2, scenarios.evaluators.cost_ecbs(
                self.env, starts_2, goals_2, ignore_cache=True)
        )
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
                self.env, starts_invalid, goals_invalid, ignore_cache=True)
        )
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
                self.env, starts_timeout, goals_timeout, ignore_cache=True)
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
                self.env, starts_vertex, goals_vertex, ignore_cache=True)
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
                self.env, starts_edge, goals_edge, ignore_cache=True)
        )

        # unsolvable
        starts_unsolv = np.array([
            [1, 0]  # obstacle
        ])
        goals_unsolv = np.array([
            [1, 2]  # obstacle
        ])
        self.assertEqual(
            scenarios.evaluators.INVALID, scenarios.evaluators.blocks_ecbs(
                self.env, starts_unsolv, goals_unsolv, ignore_cache=True)
        )

    def test_cost_independant(self):
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
            4, scenarios.evaluators.cost_independant(
                self.env, starts_4, goals_4, ignore_cache=True)
        )
        # agents that don't collide
        starts_2 = np.array([
            [0, 0],
            [2, 0]
        ])
        goals_2 = np.array([
            [0, 2],
            [2, 2]
        ])
        self.assertAlmostEqual(
            2, scenarios.evaluators.cost_independant(
                self.env, starts_2, goals_2, ignore_cache=True)
        )
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
            scenarios.evaluators.cost_independant(
                self.env, starts_invalid, goals_invalid, ignore_cache=True)
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
                self.env, starts_4_5, goals_4_5, ignore_cache=True)
        )

    def test_cost_sim_decentralized_random_no_collision(self):
        # agents that don't collide
        starts_2 = np.array([
            [0, 0],
            [2, 0]
        ])
        goals_2 = np.array([
            [0, 2],
            [2, 2]
        ])
        self.assertAlmostEqual(
            2, scenarios.evaluators.cost_sim_decentralized_random(
                self.env, starts_2, goals_2, ignore_cache=True)
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
                self.env, starts_unsolv, goals_unsolv, ignore_cache=True)
        )

    def test_cost_sim_decentralized_random_deadlocks(self):
        # trying to make deadlocks ...
        random.seed(1)  # this some times times out on total random
        starts_deadlocks = np.array([
            [0, 0],
            [0, 1],
            [0, 2],
            [2, 0],
            [2, 1],
            [2, 2]
        ])
        goals_deadlocks = np.array([
            [2, 0],
            [2, 1],
            [2, 2],
            [0, 0],
            [0, 1],
            [0, 2]
        ])
        self.assertEqual(
            scenarios.evaluators.INVALID,
            scenarios.evaluators.cost_sim_decentralized_random(
                self.env, starts_deadlocks, goals_deadlocks, ignore_cache=True)
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
