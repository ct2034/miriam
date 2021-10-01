#!/usr/bin/env python3
import random
from math import exp, isclose, log
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from definitions import INVALID, SCENARIO_TYPE
from matplotlib import pyplot as plt
from scenarios.evaluators import cost_ecbs
from scenarios.generators import tracing_pathes_in_the_dark
from sim.decentralized.iterators import IteratorType
from sim.decentralized.policy import (FirstThenRaisingPolicy,
                                      LearnedRaisingPolicy, Policy,
                                      PolicyCalledException, PolicyType,
                                      QLearningPolicy)
from sim.decentralized.runner import (has_exception, run_a_scenario,
                                      to_agent_objects,
                                      will_they_collide_in_scen)
from tools import ProgressBar
from torch.nn import Linear
from torch_geometric.data import Data
from torch_geometric.nn import (GCNConv, global_add_pool, global_max_pool,
                                global_mean_pool)


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
        # accumulate costs between sim steps
        self.costs_so_far = 0.

    def start(self) -> Data:
        scenario_result = run_a_scenario(
            self.env, self.agents, False,
            IteratorType.BLOCKING1,
            pause_on=PolicyCalledException,  # type: ignore
            ignore_finished_agents=False)
        if not has_exception(scenario_result):
            return None  # already finished
        (average_time, _, _, _, exception
         ) = scenario_result
        self.agent_first_raised = self.ids.index(
            exception.policy.a.id)
        self.agent_in_collision = self.ids.index(
            exception.id_coll)
        self.costs_so_far += average_time
        state = exception.get_agent_state()
        return state

    def step(self, action) -> Tuple[Optional[Data], float]:
        """based on current state of the scenario, take this action and
        continue to run it until the next policy call or finish"""
        # cost if simulation was unsuccessfull:
        UNSUCCESSFUL_COST = -2.
        # take step also on other agent
        # TODO: for double the data we would need to also get the state of the
        # other agent, here
        first_raised_policy: Policy = FirstThenRaisingPolicy(
            self.agents[self.agent_first_raised], int(action))
        in_collision_policy: Policy = FirstThenRaisingPolicy(
            self.agents[self.agent_in_collision], int(not bool(action)))
        self.agents[self.agent_first_raised].policy = first_raised_policy
        self.agents[self.agent_in_collision].policy = in_collision_policy
        # continue to run
        scenario_result = run_a_scenario(
            self.env, self.agents, False,
            IteratorType.BLOCKING1,
            pause_on=PolicyCalledException,  # type: ignore
            ignore_finished_agents=False)
        # return either new state or results
        if has_exception(scenario_result):  # not done
            (average_time, _, _, _, exception
             ) = scenario_result
            # reset policies
            first_raised_policy = LearnedRaisingPolicy(
                self.agents[self.agent_first_raised])
            in_collision_policy = LearnedRaisingPolicy(
                self.agents[self.agent_in_collision])
            self.agents[self.agent_first_raised].policy = first_raised_policy
            self.agents[self.agent_in_collision].policy = in_collision_policy
            # record new agent ids
            self.agent_first_raised = self.ids.index(
                exception.policy.a.id)
            self.agent_in_collision = self.ids.index(
                exception.id_coll)
            # record current state
            state: Optional[Data] = (
                exception.get_agent_state())
            self.costs_so_far += average_time
            reward = 0.
        elif not has_exception(scenario_result):  # done
            (average_time, _, _, _, successful
             ) = scenario_result
            if successful:
                cost_decen = self.costs_so_far + average_time
                reward = self.ecbs_cost - cost_decen
            else:
                reward = UNSUCCESSFUL_COST
            # state does not matter now
            state = None
        else:
            raise RuntimeError()
        return state, reward


def make_useful_scenarios(n: int, seed) -> List[Scenario]:
    scenarios: List[Scenario] = []
    if n > 1:
        pb = ProgressBar("Data Generation", n, 5)
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
    def __init__(self, num_node_features, num_actions,
                 hidden_channels) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels*3, num_actions)

    def forward(self, data: Data):
        x = data.x
        edge_index = data.edge_index
        pos = data.pos
        # everything is one batch
        batch = torch.tensor([0 for _ in range(x.shape[0])], dtype=torch.int64)

        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.conv3(x, edge_index)
        x = x.relu()

        # 2. Readout layer
        x = torch.cat((
            global_mean_pool(x, batch),
            global_max_pool(x, batch),
            global_add_pool(x, batch)
        ), 1)

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.lin(x)
        return x

    def get_action_and_q_value_training(self, state: Data):
        self.train()
        q = self.forward(state)
        action = int(torch.argmax(q))
        return action, q

    def copy_to(self, other_fn):
        return other_fn.load_state_dict(self.state_dict())


def sample_random_minibatch(n: int, d, qfun_hat, gamma: float):
    n = min(n, len(d))
    memory_tuples = random.choices(d, k=n)
    training_batch = []
    for mt in memory_tuples:
        (state, action, reward, next_state) = mt
        if next_state is not None:  # not the end
            # 11
            assert reward == 0.
            qvals = qfun_hat(next_state)
            y = gamma * torch.max(qvals)
        else:
            # 10
            y = reward
        training_sample = (state, action, y)
        training_batch.append(training_sample)
    return training_batch


def train(training_batch, qfun, optimizer):
    qfun.train()
    losss = torch.zeros(len(training_batch))
    for i_b, tb in enumerate(training_batch):
        optimizer.zero_grad()
        (state, action, y) = tb
        qvals = qfun(state)
        loss = (y - qvals[0, action]) ** 2
        loss.backward()
        optimizer.step()
        losss[i_b] = loss
    return torch.mean(losss)


def evaluate(data_test: List[Scenario], qfun):
    average_times = []
    successfuls = []
    suboptimalities = []
    for scenario in data_test:
        for a in scenario.agents:
            # reset all agents
            a.back_to_the_start()
            a.policy = QLearningPolicy(a)
            a.policy.set_qfun(qfun)
        res = run_a_scenario(a.env, scenario.agents,
                             False, IteratorType.BLOCKING1,
                             ignore_finished_agents=False)
        assert not has_exception(res)
        (average_time, _, _, _, successful) = res
        successfuls.append(successful)
        if successful:
            average_times.append(average_time)
            suboptimality = average_time - scenario.ecbs_cost
            suboptimalities.append(suboptimality)
            if (suboptimality < 0 and
                    not isclose(suboptimality, 0, abs_tol=1E-5)):
                assert suboptimality >= 0
    return (np.mean(np.array(average_times)),
            np.mean(np.array(successfuls)),
            np.mean(np.array(suboptimalities)))


def q_learning(n_episodes: int, eps_start: float,
               c: int, gamma: float, n_training_batch: int):
    """Q-learning with experience replay
    pseudocode from https://github.com/diegoalejogm/deep-q-learning
    :param n_episodes: how many episodes to simulate
    :param eps_start: initial epsilon value
    :param eps_decay: epsilon decay factor per episode
    :param c: reset qfun_hat every c episodes
    :param gamma: discout factor for future rewards
    :param n_training_batch: size of training minibatch
    """
    time_limit = 100

    # epsilon paramters
    eps_end = .01
    eps_alpha = -1 * log(eps_end / eps_start) / n_episodes

    n_data_test = 50
    data_test = make_useful_scenarios(n_data_test, n_episodes * 11)

    qfun = Qfunction(6, 2, 16)
    qfun_hat = Qfunction(6, 2, 16)
    qfun.copy_to(qfun_hat)

    # replay memory
    # (state, action, reward, next state)
    d: List[Tuple[Data, int, float, Optional[Data]]] = []
    d_max_len = 200

    # optimizer
    optimizer = torch.optim.Adam(qfun.parameters())

    # stats
    epsilons = []
    rewards = []
    losss = []
    eval_succ = []
    eval_time = []
    eval_subopt = []
    stat_every = max(1, int(n_episodes / 100))
    i_o = 0  # count optimizations

    pb = ProgressBar("Epochs", n_episodes, 5)
    # 1
    for i_e in range(n_episodes):
        # 2
        [scenario] = make_useful_scenarios(1, i_e * 10)
        epsilon = eps_start * exp(-eps_alpha * i_e)
        state = scenario.start()
        next_state = None
        # 3
        for i_t in range(time_limit):
            if state is not None:  # episode has not ended
                rand = random.random()
                if rand > epsilon:  # exploration
                    # 4
                    action = int(random.random())
                else:  # exploitation
                    # 5
                    action, _ = qfun.get_action_and_q_value_training(state)
                # 6
                next_state, reward = scenario.step(action)
                # 7
                # if next_state is None:  # agents reached their goals
                # 8 store in replay memory
                memory_tuple = (
                    state,
                    action,
                    reward,
                    next_state
                )
                if len(d) < d_max_len:
                    d.append(memory_tuple)
                else:
                    d[random.randint(0, d_max_len-1)] = memory_tuple
                # 9
                training_batch = sample_random_minibatch(
                    n_training_batch, d, qfun_hat, gamma)
                # 12
                loss = train(training_batch, qfun, optimizer)
                i_o += 1
                # 13
                if i_o % c == 0:
                    qfun.copy_to(qfun_hat)
                # for next round
                del state
                state = next_state
            else:  # episode has ended
                break
        if i_e % stat_every == 0:
            rewards.append(reward)
            losss.append(float(loss))
            epsilons.append(epsilon)
            # evaluation
            time, succ, subopt = evaluate(data_test, qfun)
            eval_time.append(time)
            eval_succ.append(succ)
            eval_subopt.append(subopt)
        del scenario
        del state
        del next_state
        pb.progress()
    pb.end()

    # print stats
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax1.plot(epsilons, label="epsilon")
    ax1.legend()
    ax2.plot(rewards, label="reward")
    ax2.plot(losss, label="loss")
    ax2.legend()
    ax3.plot(eval_time, label="eval_time")
    ax3.plot(eval_succ, label="eval_succ")
    ax3.plot(eval_subopt, label="eval_subopt")
    ax3.legend()
    plt.savefig('statistics.png')
    plt.show()


if __name__ == "__main__":
    q_learning(
        n_episodes=1000,
        eps_start=.9,
        c=100,
        gamma=.99,
        n_training_batch=100
    )
