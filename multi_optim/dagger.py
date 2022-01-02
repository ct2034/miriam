import logging
from typing import Optional, Tuple

import numpy as np
import scenarios
from definitions import INVALID, SCENARIO_RESULT
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
OBSERVATION = Tuple[Data, int]
MAX_STEPS = 10
N_LEARN_MAX = 100


def get_input_data_from_observation(
        observation: OBSERVATION) -> MODEL_INPUT:
    data, node, big_from_small = observation
    x = data.x
    edge_index = data.edge_index
    pos = data.pos
    out = (x, edge_index, pos, node)
    return out


class ScenarioState():
    """A mapf scenario in a given state that can be executed step-wise"""

    def __init__(self, graph, starts, goals, i_agent_to_consider,
                 env_nx, model) -> None:
        self.graph = graph
        self.starts = starts
        self.goals = goals
        self.i_agent_to_consider = i_agent_to_consider
        self.finished = False
        self.agents = to_agent_objects(graph, starts, goals, env_nx=env_nx)
        self.model = model
        for i in range(len(self.agents)):
            if i != self.i_agent_to_consider:
                self.agents[i].policy = EdgePolicy(self.agents[i], model)
            else:
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
        if has_exception(scenario_result):
            self.finished = True
        return ScenarioState(
            graph=self.graph,
            starts=[a.pos for a in self.agents],  # this is the new start
            goals=self.goals,
            i_agent_to_consider=self.i_agent_to_consider,
            env_nx=env_to_nx(self.graph),
            model=self.model)

    def observe(self) -> Optional[OBSERVATION]:
        """Return the observation of the current state, None if finished"""
        if self.finished:
            return None
        return agents_to_data(self.agents, self.i_agent_to_consider)

    def step(self, action: ACTION):
        """Perform the given action and return the new state"""
        self.agents[self.i_agent_to_consider].policy = EdgeThenRaisingPolicy(
            self.agents[self.i_agent_to_consider], action)
        return self.run()


class DaggerStrategy():
    """Implementation of DAgger 
    (https://proceedings.mlr.press/v15/ross11a.html)"""

    def __init__(self, model, graph, n_episodes, n_agents, rng):
        self.d = set()
        self.model = model
        self.graph = graph
        self.n_episodes = n_episodes
        self.n_agents = n_agents
        self.env_nx = env_to_nx(graph)
        self.rng = rng

        # Initialize the policy on first run
        if self.model is None:
            self.model = EdgePolicyModel()

    def sample_trajectory(self, rng, max_steps=MAX_STEPS):
        """Sample a trajectory using the given policy."""
        solvable = False
        while not solvable:
            starts = rng.sample(self.graph.nodes(), self.n_agents)
            goals = rng.sample(self.graph.nodes(), self.n_agents)
            # is this solvable?
            paths = scenarios.solvers.cached_cbsr(
                self.graph, starts, goals, radius=RADIUS, timeout=int(TIMEOUT*.9))
            if paths != INVALID:
                solvable = True

        i_a = rng.randrange(self.n_agents)
        s = ScenarioState(self.graph, starts, goals,
                          i_a, self.env_nx, self.model)

        # Sample initial state
        state = s.run()
        states = []
        for i_s in range(max_steps):
            try:
                observation = state.observe()
                if observation is None:
                    break
                scores, targets = self.model(
                    *(get_input_data_from_observation(observation)))
                action = int(targets[scores.argmax()])
                # Take action
                state = state.step(action)
                states.append(state)
            except RuntimeError as e:
                logger.warning("RuntimeError: {}".format(e))
                break
        return states

    def get_optimal_action(self, state):
        """Get the optimal action for the given state."""
        return get_optimal_edge(state.agents, state.i_agent_to_consider)

    def run_dagger(self):
        """Run the DAgger algorithm."""
        loss_s = []
        for i_e in range(self.n_episodes):
            # sample new data
            states = self.sample_trajectory(self.rng)
            for state in states:
                try:
                    action = self.get_optimal_action(state)
                    self.d.add((state, action))
                except RuntimeError as e:
                    logger.warning("RuntimeError: {}".format(e))

            # learn
            ds = self.rng.sample(self.d, min(len(self.d), N_LEARN_MAX))
            loss = self.model.learn(
                [get_input_data_from_observation(s.observe()) for s, _ in ds],
                [a for _, a in ds])
            loss_s.append(loss)

        return self.model, np.mean(loss_s)
