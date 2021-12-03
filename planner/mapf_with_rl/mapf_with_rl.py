#!/usr/bin/env python3
import json
import logging
import multiprocessing as mp
import random
from math import exp, log
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from definitions import INVALID, SCENARIO_TYPE
from planner.mapf_with_rl.mapf_with_rl_plot import (
    make_plot_from_json, make_plots_for_all_files_in_results_dir,
    make_summary_plot_for_all_files_in_results_dir)
from scenarios.generators import (GENERATOR_TYPE, arena_with_crossing,
                                  building_walls, corridor_with_passing,
                                  random_fill, tracing_pathes_in_the_dark)
from scenarios.solvers import cached_ecbs
from sim.decentralized.agent import get_t_from_env
from sim.decentralized.iterators import IteratorType
from sim.decentralized.policy import (FirstThenRaisingPolicy,
                                      InverseQLearningPolicy,
                                      PolicyCalledException, PolicyType,
                                      QLearningPolicy)
from sim.decentralized.runner import (has_exception, run_a_scenario,
                                      to_agent_objects, will_scenario_collide)
from tools import ProgressBar
from torch.nn import Linear
from torch_geometric.data import Data
from torch_geometric.nn import (GCNConv, global_add_pool, global_max_pool,
                                global_mean_pool)

ctx = mp.get_context('spawn')
N_PROCESSES = min(8, mp.cpu_count())


def get_ecbs_cost(scen: SCENARIO_TYPE, ignore_finished_agents: bool):
    env, starts, goals = scen
    data = cached_ecbs(
        env=env,
        starts=starts,
        goals=goals,
        suboptimality=1.0,
        timeout=5,
        disappear_at_goal=ignore_finished_agents)
    if data is INVALID:
        return INVALID
    cost = data['statistics']['cost']
    return cost / len(starts)


class Scenario(object):
    def __init__(self, this_data: SCENARIO_TYPE, ignore_finished_agents: bool,
                 hop_dist: int, rng: random.Random) -> None:
        super().__init__()
        # the scenario
        self.env, self.starts, self.goals = this_data
        # configuration
        self.ignore_finished_agents = ignore_finished_agents
        self.hop_dist = hop_dist
        # trying to solve with ecbs
        self.ecbs_cost = get_ecbs_cost(
            this_data, ignore_finished_agents)
        self.useful = False
        if self.ecbs_cost != INVALID:
            self.useful = True
        if not self.useful:
            return
        # if useful, we go on
        self.agents = to_agent_objects(
            self.env, self.starts, self.goals,
            policy=PolicyType.LEARNED_RAISING, rng=rng)
        if self.agents is INVALID:
            self.useful = False
            return
        # initialization to save state
        # the agent that got its policy called first
        self.agent_first_raised: Optional[int] = None
        # agent that the above was in collision with
        self.agent_in_collision: Optional[int] = None
        # agents own ids related to their index, here
        self.ids = list(map(lambda a: a.id, self.agents))
        # accumulate costs between sim steps
        self.costs_so_far = 0.
        # cost if simulation was unsuccessful:
        self.UNSUCCESSFUL_COST = -.6
        # maximum number of steps, i.e. timeout
        self.TIME_LIMIT = get_t_from_env(self.env) - 1

    def start(self) -> Data:
        state, reward = self._run()
        if state is None:
            print("state is None right after start")
        return state

    def step(self, action) -> Tuple[Optional[Data], float]:
        """based on current state of the scenario, take this action and
        continue to run it until the next policy call or finish"""
        # take step also on other agent
        # TODO: for double the data we would need to also get the state of the
        # other agent, here
        first_raised_policy = FirstThenRaisingPolicy(
            self.agents[self.agent_first_raised], int(action))
        in_collision_policy = FirstThenRaisingPolicy(
            self.agents[self.agent_in_collision], int(not bool(action)))
        assert (first_raised_policy.first_return_value !=
                in_collision_policy.first_return_value)
        self.agents[self.agent_first_raised].policy = first_raised_policy
        self.agents[self.agent_in_collision].policy = in_collision_policy
        self.agent_first_raised = None
        self.agent_in_collision = None
        state, reward = self._run()
        return state, reward

    def _run(self):
        # continue to run
        scenario_result = run_a_scenario(
            self.env, self.agents, False,
            IteratorType.BLOCKING1,
            pause_on=PolicyCalledException,  # type: ignore
            ignore_finished_agents=self.ignore_finished_agents,
            time_limit=self.TIME_LIMIT)
        # return either new state or results
        if has_exception(scenario_result):  # not done
            (average_time, _, _, _, exception
             ) = scenario_result
            # record new agent ids
            self.agent_first_raised = self.ids.index(
                exception.policy.a.id)
            self.agent_in_collision = self.ids.index(
                exception.id_coll)
            # record current state
            state: Optional[Data] = (
                exception.get_agent_state(self.hop_dist))
            self.costs_so_far += average_time
            reward = 0.
        elif not has_exception(scenario_result):  # done
            (average_time, _, _, _, successful
             ) = scenario_result
            if successful:
                cost_decen = self.costs_so_far + average_time
                reward = self.ecbs_cost - cost_decen
            else:
                reward = self.UNSUCCESSFUL_COST
            # state does not matter now
            state = None
        else:
            raise RuntimeError()
        return state, reward


def make_a_useful_scenario(ignore_finished_agents: bool, size: int,
                           n_agents: int, hop_dist: int, generator: GENERATOR_TYPE,
                           rng: random.Random, pb: Optional[ProgressBar] = None,
                           i_r: Optional[int] = None) -> Scenario:
    MAX_RETRIES = 100
    if i_r is not None:
        for _ in range(i_r):
            # changing state of rng depending on run index
            _ = rng.random()
    for _ in range(MAX_RETRIES):
        scen_data: SCENARIO_TYPE = generator(
            size,  # size
            .3,  # fill
            n_agents,  # n_agents
            rng)  # rng
        (env, starts, goals) = scen_data
        collide = will_scenario_collide(
            env, starts, goals, ignore_finished_agents)
        if collide == True:
            scen = Scenario(scen_data, ignore_finished_agents, hop_dist, rng)
            if scen.useful:
                if pb is not None and i_r is not None:
                    pb.progress(i_r)
                return scen
    raise RuntimeError(
        f"could not find a useful scenario after {MAX_RETRIES} retries.")


def proxy_make_a_useful_scenario(args):
    return make_a_useful_scenario(*args)


def make_useful_scenarios(n: int, ignore_finished_agents: bool, size: int,
                          n_agents: int, hop_dist: int,
                          generator: GENERATOR_TYPE,
                          rng: random.Random) -> List[Scenario]:
    assert generator in [arena_with_crossing, corridor_with_passing,
                         tracing_pathes_in_the_dark, building_walls, random_fill]
    if n == 1:
        try:
            scenarios = [make_a_useful_scenario(
                ignore_finished_agents, size, n_agents, hop_dist, generator,
                rng)]
        except RuntimeError:
            # retry
            scenarios = make_useful_scenarios(
                n, ignore_finished_agents, size, n_agents, hop_dist, generator, rng)
    else:
        p = ctx.Pool(N_PROCESSES)
        pb = ProgressBar("Data Generation", n, 5)
        args = [(ignore_finished_agents, size, n_agents, hop_dist, generator,
                 rng, pb, i_r) for i_r in range(n)]
        scenarios = p.map(proxy_make_a_useful_scenario, args)
        pb.end()
        p.close()
        p.join()
    return scenarios


class Qfunction(torch.nn.Module):
    def __init__(self, num_node_features, num_actions,
                 hidden_channels) -> None:
        super().__init__()
        INIT_STD = .03
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        # self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels*3, num_actions)

        for net in [self.conv1, self.conv2]:
            torch.nn.init.normal_(net.bias, 0, INIT_STD)

        for net in [self.conv1.lin, self.conv2.lin, self.lin]:
            torch.nn.init.normal_(net.weight, 0, INIT_STD)
            if net.bias is not None:
                torch.nn.init.normal_(net.bias, 0, INIT_STD)

    def forward(self, data: Data):
        x = data.x
        edge_index = data.edge_index
        pos = data.pos
        # everything is one batch
        batch = torch.tensor([0 for _ in range(x.shape[0])], dtype=torch.int64)

        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv2(x, edge_index)
        x = x.relu()
        # x = F.dropout(x, p=0.1, training=self.training)
        # x = self.conv3(x, edge_index)
        # x = x.relu()

        # 2. Readout layer
        x = torch.cat((
            global_mean_pool(x, batch),
            global_max_pool(x, batch),
            global_add_pool(x, batch)
        ), 1)

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.lin(x)
        return x

    def get_action_and_q_value_training(self, state: Data):
        self.eval()
        q = self.forward(state)
        action = int(torch.argmax(q))
        return action, q

    def copy_to(self, other_fn):
        return other_fn.load_state_dict(self.state_dict())


def sample_random_minibatch(n: int, d, qfun_hat, gamma: float, rng: random.Random):
    n = min(n, len(d))
    memory_tuples = rng.choices(d, k=n)
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
        (state, action, y) = tb
        qvals = qfun(state)
        loss = (y - qvals[0, action]) ** 2
        losss[i_b] = loss
    mean_loss = torch.mean(losss)
    optimizer.zero_grad()
    mean_loss.backward()
    optimizer.step()
    return mean_loss


def evaluate(data_test: List[Scenario], qfun, ignore_finished_agents: bool, hop_dist: int,
             inverse: bool):
    successful_s = []
    regret_s = []
    qfun.eval()
    for scenario in data_test:
        for a in scenario.agents:
            # reset all agents
            a.back_to_the_start()
            if inverse:
                a.policy = InverseQLearningPolicy(a, hop_dist)
            else:
                a.policy = QLearningPolicy(a, hop_dist)
            a.policy.set_qfun(qfun)
        res = run_a_scenario(a.env, scenario.agents,
                             False, IteratorType.BLOCKING1,
                             ignore_finished_agents=ignore_finished_agents)
        assert not has_exception(res)
        (average_time, _, _, _, successful) = res
        successful_s.append(successful)
        if successful:
            regret = average_time - scenario.ecbs_cost
            regret_s.append(regret)
            # if (regret < 0 and
            #         not isclose(regret, 0, abs_tol=1E-5)):
            #     logging.warning(f"regret: {regret}")
    mean_successful = np.mean(np.array(successful_s))
    mean_regret = np.mean(np.array(regret_s))
    return (mean_successful, mean_regret)


def q_learning(n_episodes: int, eps_start: float,
               c: int, gamma: float, n_training_batch: int,
               test_scenarios: Dict[str, List[Scenario]],
               ignore_finished_agents: bool, hop_dist: int, seed: int, name: str):
    """Q-learning with experience replay
    pseudocode from https://github.com/diegoalejogm/deep-q-learning
    :param n_episodes: how many episodes to simulate
    :param eps_start: initial epsilon value
    :param eps_decay: epsilon decay factor per episode
    :param c: reset qfun_hat every c episodes
    :param gamma: discout factor for future rewards
    :param n_training_batch: size of training minibatch
    :param data_test: dict of test scenarios
    :param ignore_finished_agents: wether or not to ignore agents at their
           goal pose
    :param hop_dist: how big the agent state graph is (from current pos)
    :param seed: for the random number generator
    :param name: to save the final plot under ({name}.png)
    """
    # random number generator
    rng = random.Random(seed)
    torch.manual_seed(seed)

    # epsilon paramters
    eps_end = .01
    eps_alpha = -1 * log(eps_end / eps_start) / n_episodes

    # init qfun
    qfun = Qfunction(9, 2, 16)
    qfun_hat = Qfunction(9, 2, 16)
    qfun.copy_to(qfun_hat)

    # replay memory
    # (state, action, reward, next state)
    d: List[Tuple[Data, int, float, Optional[Data]]] = []
    d_max_len = n_training_batch * 100

    training_start = d_max_len/2

    # optimizer
    optimizer = torch.optim.Adam(
        qfun.parameters(),
        lr=1e-3)  # default

    # size changes
    training_sizes = {
        .0: (3, 2, [arena_with_crossing]),
        .4: (5, 4, [arena_with_crossing]),
        .6: (7, 6, [arena_with_crossing])
        # .8: (10, 2, [arena_with_crossing])
    }

    # stats
    stats = {name: ([], []) for name in [
        "epsilons",
        "rewards",
        "loss",
        "max_q",
        "min_q",
        "d_fill",
        "timesteps",
    ] + [f"eval_success_{k}" for k in test_scenarios.keys()
         ] + [f"eval_regret_{k}" for k in test_scenarios.keys()
              ] + [f"eval_success_inv_{k}" for k in test_scenarios.keys()
                   ] + [f"eval_regret_inv_{k}" for k in test_scenarios.keys()
                        ]}  # type: Dict[str, Tuple[List[float], List[float]]]
    loss = 0
    stat_every = max(1, int(n_episodes / 1000))
    eval_every = max(1, int(n_episodes / 20))
    i_o = 0  # count optimizations

    pb = ProgressBar(f"Run {name}", n_episodes, 1)
    # 1
    for i_e in range(n_episodes):
        # 2
        for progress in sorted(training_sizes.keys()):
            if i_e >= int(n_episodes * progress):
                size, n_agents, generators = training_sizes[progress]
                generator = rng.choice(generators)
        [scenario] = make_useful_scenarios(
            1, ignore_finished_agents, size, n_agents, hop_dist, generator, rng)
        # TODO only decrease after training_start
        epsilon = eps_start * exp(-eps_alpha * i_e)
        state = scenario.start()
        next_state = None
        # 3
        for i_t in range(scenario.TIME_LIMIT):
            if state is not None:  # episode has not ended
                rand = rng.random()
                if rand > epsilon:  # exploration
                    # 4
                    action = int(rng.random())
                else:  # exploitation
                    # 5
                    action, [qvals] = qfun.get_action_and_q_value_training(
                        state)
                # 6
                next_state, reward = scenario.step(action)
                if i_t == scenario.TIME_LIMIT-1 and next_state is not None:  # time out
                    reward = scenario.UNSUCCESSFUL_COST
                    next_state = None
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
                    d[i_e % d_max_len] = memory_tuple
                if len(d) > training_start:
                    # 9
                    training_batch = sample_random_minibatch(
                        n_training_batch, d, qfun_hat, gamma, rng)
                    # 12
                    loss = train(training_batch, qfun, optimizer)
                    i_o += 1
                # for next round
                del state
                state = next_state
            else:  # episode has ended
                break
        # 13
        if i_e % c == 0:
            qfun.copy_to(qfun_hat)
        if i_e % stat_every == 0:
            stats["d_fill"][0].append(i_e)
            stats["d_fill"][1].append(float(len(d))/d_max_len)
            stats["rewards"][0].append(i_e)
            stats["rewards"][1].append(reward)
            stats["loss"][0].append(i_e)
            stats["loss"][1].append(float(loss))
            stats["epsilons"][0].append(i_e)
            stats["epsilons"][1].append(epsilon)
            stats["max_q"][0].append(i_e)
            stats["max_q"][1].append(float(max(qvals)))
            stats["min_q"][0].append(i_e)
            stats["min_q"][1].append(float(min(qvals)))
            stats["timesteps"][0].append(i_e)
            stats["timesteps"][1].append(i_t)
        if (i_e == 0 or
            i_e == n_episodes - 1 or
                (i_e % eval_every == 0 and len(d) > training_start)):
            # evaluation qfun
            for set_name, test_set in test_scenarios.items():
                success, regret = evaluate(
                    test_set, qfun, ignore_finished_agents, hop_dist,
                    inverse=False)
                stats[f"eval_success_{set_name}"][0].append(i_e)
                stats[f"eval_success_{set_name}"][1].append(success)
                stats[f"eval_regret_{set_name}"][0].append(i_e)
                stats[f"eval_regret_{set_name}"][1].append(regret)
                inv_success, inv_regret = evaluate(
                    test_set, qfun, ignore_finished_agents, hop_dist,
                    inverse=True)
                stats[f"eval_success_inv_{set_name}"][0].append(i_e)
                stats[f"eval_success_inv_{set_name}"][1].append(inv_success)
                stats[f"eval_regret_inv_{set_name}"][0].append(i_e)
                stats[f"eval_regret_inv_{set_name}"][1].append(inv_regret)
                print(f"evaluation test set {set_name} in {name}:\n" +
                      f"  success: {success:.2f}\n" +
                      f"  regret: {regret:.2f}\n" +
                      f"  inv success: {inv_success:.2f}\n" +
                      f"  inv regret: {inv_regret:.2f}")
        del scenario
        del state
        del next_state
        pb.progress()
    pb.end()
    with open(f"planner/mapf_with_rl/results/{name}.json", "w") as f:
        json.dump(stats, f)
    torch.save(qfun.state_dict(), f"planner/mapf_with_rl/results/{name}.pt")
    return stats


def proxy_q_learning(kwargs):
    return q_learning(**kwargs)


if __name__ == "__main__":
    logging.getLogger(
        "sim.decentralized.runner").setLevel(logging.ERROR)

    # debug run
    logging.getLogger("sim.decentralized.runner").setLevel(logging.DEBUG)
    data_test = {"random_fill": make_useful_scenarios(3, True, 4, 3, 3,
                                                      random_fill, random.Random(0)),
                 "tracing_pathes_in_the_dark": make_useful_scenarios(3, True, 8, 6, 3,
                                                                     tracing_pathes_in_the_dark, random.Random(0)),
                 "corridor_with_passing": make_useful_scenarios(3, True, 12, 2, 3,
                                                                corridor_with_passing, random.Random(0)),
                 "arena_with_crossing": make_useful_scenarios(3, True, 4, 3, 3,
                                                              arena_with_crossing, random.Random(0))}
    stats = q_learning(
        n_episodes=20,
        eps_start=.9,
        c=2,
        gamma=.9,
        n_training_batch=2,
        test_scenarios=data_test,
        ignore_finished_agents=True,
        hop_dist=4,
        seed=0,
        name=f"debug"
    )
    make_plot_from_json("debug")

    # real run
    logging.getLogger("sim.decentralized.runner").setLevel(logging.INFO)
    n_data_test = 100
    test_scenarios = {
        "arena_with_crossing_2agents": make_useful_scenarios(
            n_data_test, True, 8, 2, 3, arena_with_crossing, random.Random(0)),
        "arena_with_crossing_4agents": make_useful_scenarios(
            n_data_test, True, 8, 4, 3, arena_with_crossing, random.Random(0)),
        "arena_with_crossing_6agents": make_useful_scenarios(
            n_data_test, True, 8, 6, 3, arena_with_crossing, random.Random(0)),
    }

    n_runs = 8
    kwargs = [
        {
            "n_episodes": int(10E3),
            "eps_start": .9,
            "c": 100,
            "gamma": .9,
            "n_training_batch": 64,
            "test_scenarios": test_scenarios,
            "ignore_finished_agents": True,
            "hop_dist": 4,
            "seed": i_r,
            "name": f"run{i_r}_increasing"
        } for i_r in range(n_runs)
    ]

    p = ctx.Pool(N_PROCESSES)
    results = p.map(proxy_q_learning, kwargs)
    p.close()
    p.join()

    # make plots
    make_summary_plot_for_all_files_in_results_dir()
    make_plots_for_all_files_in_results_dir()
