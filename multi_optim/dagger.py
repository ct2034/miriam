import copy
import logging
import os
import pickle
import tracemalloc
from random import Random
from typing import Dict, List, Optional, Tuple

import numpy as np
import scenarios
import scenarios.solvers
import tools
import torch
from definitions import INVALID, SCENARIO_RESULT
from humanfriendly import format_size
from planner.policylearn.edge_policy import (MODEL_INPUT, EdgePolicyDataset,
                                             EdgePolicyModel)
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
from torch_geometric.loader import DataLoader

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
        self.paths_out = []  # type: List[List[int]]

    def run(self):
        """Start the scenario and return the initial state"""
        paths_out = self.paths_out
        scenario_result: SCENARIO_RESULT = run_a_scenario(
            env=self.graph,
            agents=self.agents,
            plot=False,
            iterator=IteratorType.EDGE_POLICY2,
            pause_on=PolicyCalledException,
            paths_out=paths_out)
        self.is_agents_to_consider = None
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


def make_a_state_with_an_upcoming_decision(
        graph, n_agents, env_nx, model, rng) -> ScenarioState:
    useful = False
    while not useful:
        starts = rng.sample(graph.nodes(), n_agents)
        goals = rng.sample(graph.nodes(), n_agents)
        state = ScenarioState(graph, starts, goals, env_nx, model)
        state.run()
        if not state.finished:
            try:
                _ = get_optimal_edge(state.agents, 0)
                useful = True
            except RuntimeError:
                pass
    return state


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
    state.run()

    # Sample states
    these_ds = []
    for i_s in range(max_steps):
        try:
            if state.finished:
                break
            observations = state.observe()
            actions: Dict[int, ACTION] = {}
            for i_a, (d, bfs) in observations.items():
                # find actions to take using the policy
                actions[i_a] = model.predict(d.x, d.edge_index, bfs)
                # observation, action pairs for learning
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
                 batch_size, optimizer, prefix, rng):
        self.model = model
        self.graph = self._add_self_edges_to_graph(graph)
        self.n_episodes = n_episodes
        self.n_agents = n_agents
        self.batch_size = batch_size
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

    def run_dagger(self, pool, epds):
        """Run the DAgger algorithm."""
        loss_s = []
        data_lengths = []

        logger.debug("Memory usage, current: " +
                     str(format_size(tracemalloc.get_traced_memory()[0])) +
                     " peak: " +
                     str(format_size(tracemalloc.get_traced_memory()[1])))

        model_copy = EdgePolicyModel()
        model_copy.load_state_dict(copy.deepcopy(self.model.state_dict()))
        model_copy.eval()

        params = [(s, self.graph, self.n_agents, self.env_nx,
                   model_copy) for s in self.rng.sample(
            range(2**32), k=self.n_episodes)]
        generation_hash = tools.hasher([], {
            "seeds": [p[0] for p in params],
            "self.graph": self.graph,
            "self.n_agents": self.n_agents,
            "model": model_copy
            # env_nx is not needed, because it depends on the graph
        })
        new_fname = self._get_path_data(generation_hash)
        epds.add_file(new_fname)

        # only create file if this data does not exist
        if os.path.exists(new_fname):
            pass
        else:
            results_s = pool.imap_unordered(
                sample_trajectory_proxy, params)
            new_ds = []
            for results in results_s:
                new_ds.extend(results)
            with open(new_fname, "wb") as f:
                pickle.dump(new_ds, f)

        # learn
        loader = DataLoader(epds, batch_size=self.batch_size, shuffle=True)
        loss_s = []
        for i_b, batch in enumerate(loader):
            loss = self.model.learn(batch, self.optimizer)
            loss_s.append(loss)

        # statistics
        len_new_data = sum(
            1 if generation_hash in x[0] else 0 for x in epds.lookup)
        new_data_percentage = len_new_data / len(epds)

        if len(loss_s) == 0:
            loss_s = [0]
        return (self.model, np.mean(loss_s), new_data_percentage,
                epds, len(epds))
