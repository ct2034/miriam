import unittest

import numpy as np
import torch
from planner.mapf_with_rl.mapf_with_rl import (Qfunction, Scenario,
                                               make_useful_scenarios)
from torch_geometric.data import Data


class TestMapfWithRl(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName=methodName)
        self.env = np.array([
            [0, 0, 0],
            [1, 0, 1],
            [0, 0, 0]])
        self.starts = np.array([
            [0, 0],
            [0, 2]])
        self.goals = np.array([
            [2, 0],
            [2, 1]])  # this will block

    def test_scenario_init_useful(self):
        s = Scenario((self.env, self.starts, self.goals))
        self.assertTrue(s.useful)

    def test_scenario_init_not_useful(self):
        env_blocked = self.env.copy()
        env_blocked[1, 1] = 1
        s = Scenario((env_blocked, self.starts, self.goals))
        self.assertFalse(s.useful)

    def _data_equality(self, a: Data, b: Data):
        if not a.keys == b.keys:
            return False
        comparisons = []
        for key in a.keys:
            comparisons.append(
                str(a[key]) == str(b[key])
            )
        return all(comparisons)

    def test_scenario_running(self):
        # Different actions lead to different results.
        s0 = Scenario((self.env, self.starts, self.goals))
        first_state_0 = s0.start()
        second_state_action_0, reward_0 = s0.step(0)
        self.assertEqual(reward_0, 0)

        s1 = Scenario((self.env, self.starts, self.goals))
        self.assertEqual(s1.agent_first_raised, None)
        first_state_1 = s1.start()
        self.assertEqual(s1.agent_first_raised, 0)

        second_state_action_1, reward_1 = s1.step(1)
        self.assertEqual(reward_1, 0)
        # finished now
        self.assertEqual(second_state_action_1, None)

        self.assertTrue(self._data_equality(
            first_state_0, first_state_1))
        self.assertFalse(self._data_equality(
            first_state_0, second_state_action_0))

        # stepping again should end it
        second_state_action_0, reward_0 = s0.step(0)
        self.assertEqual(second_state_action_0, None)
        # failed scenario because agent 1 is blocking the way
        self.assertEqual(reward_0, s0.UNSUCCESSFUL_COST)

    def test_make_useful_scenarios(self):
        scenarios_0 = make_useful_scenarios(3, 0)
        self.assertEqual(len(scenarios_0), 3)
        scenarios_1 = make_useful_scenarios(3, 1)
        self.assertEqual(len(scenarios_1), 3)

        for s in scenarios_0 + scenarios_1:
            self.assertTrue(s.useful)
            s.start()

    def test_qfunction(self):
        # seeing if different qfuns give different results on different data
        torch.manual_seed(0)
        qfun_a = Qfunction(2, 3, 4)
        qfun_a.eval()
        qfun_b = Qfunction(2, 3, 4)
        qfun_b.eval()
        dummy_data_a = Data(
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            pos=None,  # currently not used
            x=torch.tensor([[1., 1.], [1., 1.]])
        )
        dummy_data_b = Data(
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            pos=None,  # currently not used
            x=torch.tensor([[1., 1.], [1., 0.]])
        )

        actions_a_a = qfun_a(dummy_data_a)
        self.assertEqual(actions_a_a.shape, (1, 3))
        actions_a_b = qfun_a(dummy_data_b)
        self.assertEqual(actions_a_a.shape, (1, 3))
        actions_b_b = qfun_b(dummy_data_b)
        self.assertEqual(actions_a_a.shape, (1, 3))
        # same qfun, different data
        self.assertTrue((actions_a_a != actions_a_b).all())
        # different qfun, same data
        self.assertTrue((actions_a_b != actions_b_b).all())

        # making qfuns the same now
        qfun_a.copy_to(qfun_b)
        actions_b_a = qfun_b(dummy_data_a)
        self.assertTrue((actions_a_a == actions_b_a).all())
