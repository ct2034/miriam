import logging
from typing import Dict, List, Optional, Tuple

from definitions import SCENARIO_RESULT
from planner.policylearn.edge_policy_graph_utils import (agents_to_data,
                                                         get_optimal_edge)
from sim.decentralized.iterators import IteratorType
from sim.decentralized.policy import (PolicyCalledException, RaisingPolicy,
                                      ValueThenRaisingPolicy)
from sim.decentralized.runner import (has_exception, run_a_scenario,
                                      to_agent_objects)
from torch_geometric.data import Data

logger = logging.getLogger(__name__)

OBSERVATION = Tuple[Data, Dict[int, int]]
ACTION = int


class ScenarioState():
    """A mapf scenario in a given state that can be executed step-wise"""

    def __init__(self, graph, starts, goals, model, radius) -> None:
        self.graph = graph
        self.starts = starts
        self.goals = goals
        self.is_agents_to_consider: Optional[List[int]] = None
        self.finished = False
        self.agents = to_agent_objects(
            graph, starts, goals, radius=radius)
        self.model = model
        if self.agents is None:
            raise RuntimeError("Error in agent generation")
        for a in self.agents:
            a.policy = RaisingPolicy(a)
        self.paths_out = []  # type: List[List[int]]

    def run(self):
        """Advance the scenario until an action is needed"""
        assert self.agents is not None
        scenario_result: SCENARIO_RESULT = run_a_scenario(
            env=self.graph,
            agents=tuple(self.agents),
            plot=False,
            iterator=IteratorType.LOOKAHEAD2,
            pause_on=PolicyCalledException,
            paths_out=self.paths_out)
        self.is_agents_to_consider = None
        if not has_exception(scenario_result):
            self.finished = True
        else:
            self.is_agents_to_consider = list(
                scenario_result[-1].agents_with_colissions)

        logger.debug("run")
        for a in self.agents:
            logger.debug(a)

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
        assert self.agents is not None
        for i_a, a in enumerate(self.agents):
            if i_a in actions.keys():
                a.policy = ValueThenRaisingPolicy(
                    a, actions[i_a])
            else:
                a.policy = RaisingPolicy(
                    a)
            a.start = a.pos
            a.back_to_the_start()
        self.run()


def make_a_state_with_an_upcoming_decision(
        graph, n_agents, model, radius: float, rng) -> ScenarioState:
    useful = False
    state = None
    while not useful:
        starts = rng.sample(graph.nodes(), n_agents)
        goals = rng.sample(graph.nodes(), n_agents)
        state = ScenarioState(graph, starts, goals, model, radius)
        state.run()
        if not state.finished:
            try:
                _ = get_optimal_edge(state.agents, 0)
                useful = True
            except RuntimeError:
                pass
    assert state
    return state
