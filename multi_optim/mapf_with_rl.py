#!/usr/bin/env python3
from typing import List, Optional, Tuple, Type, Union

import numpy as np
import torch
from definitions import INVALID, SCENARIO_RESULT, SCENARIO_TYPE
from matplotlib import pyplot as plt
from scenarios.evaluators import cost_ecbs
from scenarios.generators import tracing_pathes_in_the_dark
from scenarios.solvers import ecbs
from sim.decentralized.iterators import IteratorType
from sim.decentralized.policy import (FirstThenRaisingPolicy,
                                      LearnedRaisingPolicy, Policy,
                                      PolicyCalledException, PolicyType)
from sim.decentralized.runner import (run_a_scenario, to_agent_objects,
                                      will_they_collide_in_scen)
from tools import ProgressBar
from torch.nn import Linear
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


class Scenario(object):
    def __init__(self, this_data) -> None:
        super().__init__()
        self.env, self.starts, self.goals = this_data
        # trying to solve with ecbs
        self.ecbs_cost = cost_ecbs(
            self.env, self.starts, self.goals, timeout=5)
        self.useful = False
        if self.ecbs_cost != INVALID:
            self.useful = True
        if not self.useful:
            return
        # if useful, we go on
        self.agents = to_agent_objects(
            self.env, self.starts, self.goals,
            policy=PolicyType.LEARNED_RAISING)
        # initialization to save state
        # the agent that got its policy called first
        self.agent_first_raised: Optional[int] = None
        # agent that the above was in collision with
        self.agent_in_collision: Optional[int] = None
        # agents own ids related to their index, here
        self.ids = list(map(lambda a: a.id, self.agents))

    def start(self) -> Data:
        active_agent_exception = run_a_scenario(
            self.env, self.agents, False,
            IteratorType.BLOCKING1,
            pause_on=PolicyCalledException)  # type: ignore
        assert isinstance(active_agent_exception, PolicyCalledException)
        self.agent_first_raised = self.ids.index(
            active_agent_exception.policy.a.id)
        self.agent_in_collision = self.ids.index(
            active_agent_exception.id_coll)
        state = active_agent_exception.get_agent_state()
        return state

    def step(self, action: bool) -> Tuple[Optional[Data], float]:
        """based on current state of the scenario, take this action and
        continue to run it until the next policy call or finish"""
        # cost if simulation was unsuccessfull:
        UNSUCCESSFUL_COST = -10.
        # take step also on other agent
        # TODO: for double the data we would need to also get the state of the
        # other agent, here
        first_raised_policy: Policy = FirstThenRaisingPolicy(
            self.agents[self.agent_first_raised], int(action))
        in_collision_policy: Policy = FirstThenRaisingPolicy(
            self.agents[self.agent_in_collision], int(not action))
        self.agents[self.agent_first_raised].policy = first_raised_policy
        self.agents[self.agent_in_collision].policy = in_collision_policy
        # continue to run
        active_agent_exception_or_result = run_a_scenario(
            self.env, self.agents, False,
            IteratorType.BLOCKING1,
            pause_on=PolicyCalledException)  # type: ignore
        # return either new state or results
        if isinstance(active_agent_exception_or_result, PolicyCalledException
                      ):  # not done
            # reset policies
            first_raised_policy = LearnedRaisingPolicy(
                self.agents[self.agent_first_raised])
            in_collision_policy = LearnedRaisingPolicy(
                self.agents[self.agent_in_collision])
            self.agents[self.agent_first_raised].policy = first_raised_policy
            self.agents[self.agent_in_collision].policy = in_collision_policy
            # record new agent ids
            self.agent_first_raised = self.ids.index(
                active_agent_exception_or_result.policy.a.id)
            self.agent_in_collision = self.ids.index(
                active_agent_exception_or_result.id_coll)
            # record current state
            state: Optional[Data] = (
                active_agent_exception_or_result.get_agent_state())
            reward = 0.
        elif isinstance(active_agent_exception_or_result, tuple
                        ):  # done
            (average_time, _, _, _, successful
             ) = active_agent_exception_or_result
            if successful:
                cost_decen = average_time + 1
                reward = self.ecbs_cost - cost_decen
            else:
                reward = UNSUCCESSFUL_COST
            # state does not matter now
            print(f'reward: {reward}')
            state = None
        else:
            raise RuntimeError()
        return state, reward


def make_useful_scenarios(n: int, seed) -> List[Scenario]:
    scenarios: List[Scenario] = []
    if n > 1:
        pb = ProgressBar("Data Generation", n, 1)
    while len(scenarios) < n:
        scen_data: SCENARIO_TYPE = tracing_pathes_in_the_dark(
            size=4,
            fill=.2,
            n_agents=3,
            seed=seed)
        seed += 1
        (env, starts, goals) = scen_data
        collide, _ = will_they_collide_in_scen(env, starts, goals)
        if collide:
            scen = Scenario(scen_data)
            if scen.useful:
                if n > 1:
                    pb.progress()
                scenarios.append(scen)
    if n > 1:
        pb.end()
    return scenarios


class Qfunction(torch.nn.Module):
    def __init__(self, num_features, num_actions, hidden_channels) -> None:
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.out = Linear(hidden_channels, num_actions)

    def forward(self, data: Data) -> torch.float:
        return .5


def deep_q_learning(n_data: int, time_limit: int,
                    eps_start: float, eps_decay: float):
    epsilon = eps_start
    test_split = .1

    n_data_test = int(test_split * n_data)
    n_data_train = n_data - n_data_test
    data_test = make_useful_scenarios(n_data_test, n_data_train * 11)

    pb = ProgressBar("Epochs", n_data_train, 5)
    for i_e in range(n_data_train):
        [scenario] = make_useful_scenarios(1, i_e * 10)
        epsilon = epsilon * (1-eps_decay)
        state = scenario.start()
        qfun = Qfunction(6, 2, 32)
        action: float = qfun(state)
        print(state)
        for i_t in range(time_limit):
            state, reward = scenario.step(action)
            if reward > 0.:  # agents reached their goals
                break
            action = qfun(state)
            print(state)
        pb.progress()
    pb.end()


if __name__ == "__main__":
    deep_q_learning(10, 1000, .9, .01)
