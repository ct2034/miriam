import copy
import logging
import os
import pickle
import tracemalloc
from random import Random
from typing import Dict, List, Optional, Tuple

import numpy as np
import scenarios
import tools
import torch
from definitions import INVALID, SCENARIO_RESULT
from humanfriendly import format_size
from planner.policylearn.edge_policy import MODEL_INPUT, EdgePolicyModel
from planner.policylearn.edge_policy_graph_utils import (RADIUS, TIMEOUT,
                                                         agents_to_data,
                                                         get_optimal_edge)
from sim.decentralized.agent import env_to_nx
from sim.decentralized.iterators import IteratorType
from sim.decentralized.policy import (EdgePolicy, EdgeRaisingPolicy,
                                      EdgeThenRaisingPolicy,
                                      PolicyCalledException)
from sim.decentralized.runner import (has_exception, run_a_scenario,
                                      to_agent_objects)
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


ACTION = int
OBSERVATION = Tuple[Data, Dict[int, int]]
MAX_STEPS = 10
N_LEARN_MAX = 1000


def get_input_data_from_observation(
        observation: OBSERVATION) -> MODEL_INPUT:
    data, _ = observation
    return data


class ScenarioState():
    """A mapf scenario in a given state that can be executed step-wise"""

    def __init__(self, graph, starts, goals,
                 env_nx, model) -> None:
        self.graph = graph
        self.env_nx = env_nx
        self.starts = starts
        self.goals = goals
        self.is_agents_to_consider: Optional[List[int]] = None
        self.finished = False
        self.agents = to_agent_objects(
            graph, starts, goals, env_nx=env_nx, radius=RADIUS)
        self.model = model
        for i in range(len(self.agents)):
            self.agents[i].policy = EdgeRaisingPolicy(
                self.agents[i])

    def run(self):
        """Start the scenario and return the initial state"""
        scenario_result: SCENARIO_RESULT = run_a_scenario(
            env=self.graph,
            agents=self.agents,
            plot=False,
            iterator=IteratorType.EDGE_POLICY3,
            pause_on=PolicyCalledException)
        if not has_exception(scenario_result):
            self.finished = True
        else:
            self.is_agents_to_consider = list(
                scenario_result[-1].agents_with_colissions)

    def observe(self) -> Optional[Dict[int, OBSERVATION]]:
        """Return the observation of the current state, None if finished"""
        if self.finished:
            return None
        assert self.is_agents_to_consider is not None

        observations = {}  # type: Dict[int, OBSERVATION]
        for i_a in self.is_agents_to_consider:
            observations[i_a] = agents_to_data(
                self.agents, i_a)
        return observations

    def step(self, actions: Dict[int, ACTION]):
        """Perform the given actions and return the new state"""
        for i_a in range(len(self.agents)):
            if i_a in actions.keys():
                self.agents[i_a].policy = EdgeThenRaisingPolicy(
                    self.agents[i_a], actions[i_a])
            else:
                self.agents[i_a].policy = EdgeRaisingPolicy(
                    self.agents[i_a])
            self.agents[i_a].start = self.agents[i_a].pos
            self.agents[i_a].back_to_the_start()
        self.run()


def sample_trajectory_proxy(args):
    return sample_trajectory(*args)


def sample_trajectory(seed, graph, n_agents, env_nx,
                      model, max_steps=MAX_STEPS):
    """Sample a trajectory using the given policy."""
    rng = Random(seed)
    solvable = False
    while not solvable:
        starts = rng.sample(graph.nodes(), n_agents)
        goals = rng.sample(graph.nodes(), n_agents)
        # is this solvable?
        paths = scenarios.solvers.cached_cbsr(
            graph, starts, goals, radius=RADIUS,
            timeout=int(TIMEOUT*.9))
        if paths != INVALID:
            solvable = True

    state = ScenarioState(graph, starts, goals,
                          env_nx, model)

    # Sample initial state
    state.run()
    these_ds = []
    for i_s in range(max_steps):
        try:
            if state.finished:
                break
            observations = state.observe()
            actions: Dict[int, ACTION] = {}
            for i_a, obs in observations.items():
                # find actions to take using the policy
                d = get_input_data_from_observation(obs)
                scores, targets = model(
                    d.x, d.edge_index)
                action = int(targets[scores.argmax()])
                actions[i_a] = action
                # observation, action pairs for learning
                optimal_action = get_optimal_edge(
                    state.agents, i_a)
                d.y = optimal_action
                these_ds.append(d)
            state.step(actions)
        except RuntimeError as e:
            logger.warning("RuntimeError: {}".format(e))
            break
    return these_ds


class DaggerStrategy():
    """Implementation of DAgger
    (https://proceedings.mlr.press/v15/ross11a.html)"""

    def __init__(self, model, graph, n_episodes, n_agents,
                 optimizer, prefix, rng):
        self.model = model
        self.graph = self._add_self_edges_to_graph(graph)
        self.n_episodes = n_episodes
        self.n_agents = n_agents
        self.env_nx = env_to_nx(graph)
        self.rng = rng
        self.optimizer = optimizer
        self.prefix = prefix

    def _add_self_edges_to_graph(self, graph):
        """Add self edges to the graph."""
        for node in graph.nodes():
            if not graph.has_edge(node, node):
                graph.add_edge(node, node)
        return graph

    def _get_data_folder(self):
        return f"multi_optim/results/{self.prefix}_data"

    def _get_path_data(self, hash):
        folder = self._get_data_folder()
        if not os.path.exists(folder):
            os.makedirs(folder)
        return folder+f"/{hash}.pkl"

    def run_dagger(self, pool, data_files):
        """Run the DAgger algorithm."""
        loss_s = []
        data_lengths = []

        logger.debug("Memory usage, current: " +
                     str(format_size(tracemalloc.get_traced_memory()[0])) +
                     " peak: " +
                     str(format_size(tracemalloc.get_traced_memory()[1])))

        model_copy = EdgePolicyModel(
            self.model.num_node_features, self.model.conv_channels)
        model_copy.load_state_dict(copy.deepcopy(self.model.state_dict()))
        model_copy.eval()

        params = [(s, self.graph, self.n_agents, self.env_nx,
                   model_copy) for s in self.rng.sample(
            range(2**32), k=self.n_episodes)]
        generation_hash = tools.hasher([], {
            "seeds": [p[0] for p in params],
            "self.graph": self.graph,
            "self.n_agents": self.n_agents
            # env_nx is not needed, because it depends on the graph
        })
        new_fname = self._get_path_data(generation_hash)

        if os.path.exists(new_fname):
            logger.debug("Loading data from file")
            with open(new_fname, "rb") as f:
                new_ds = pickle.load(f)
        else:
            results_s = pool.imap_unordered(
                sample_trajectory_proxy, params
            )

            new_ds = []
            for results in results_s:
                new_ds.extend(results)
            with open(new_fname, "wb") as f:
                pickle.dump(new_ds, f)

        # learn
        # first of the current data
        data_lengths.append(len(new_ds))
        loss = self.model.learn(
            new_ds, optimizer=self.optimizer)
        if loss is not None:
            loss_s.append(loss)
        # then learn from files
        for fname in data_files:
            with open(fname, "rb") as f:
                ds = pickle.load(f)
            data_lengths.append(len(ds))
            loss = self.model.learn(
                ds, optimizer=self.optimizer)
            if loss is not None:
                loss_s.append(loss)

        # after learning add new file
        data_files.append(new_fname)

        # statistics
        new_data_perc = len(new_ds) / sum(data_lengths)

        if len(loss_s) == 0:
            loss_s = [0]
        return (self.model, np.mean(loss_s), new_data_perc,
                data_files, sum(data_lengths))
