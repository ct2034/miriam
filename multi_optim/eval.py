import logging
from random import Random
from typing import List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
from definitions import (DEFAULT_TIMEOUT_S, IDX_AVERAGE_LENGTH, IDX_SUCCESS,
                         INVALID, MAP_IMG, POS, SCENARIO_RESULT, C)
from planner.mapf_implementations.plan_cbs_roadmap import plan_cbsr
from planner.policylearn.edge_policy import BFS_TYPE, EdgePolicyModel
from pyflann import FLANN
from roadmaps.var_odrm_torch.var_odrm_torch import get_path_len, sample_points
from sim.decentralized.agent import Agent
from sim.decentralized.iterators import IteratorType
from sim.decentralized.policy import LearnedPolicy, PolicyType
from sim.decentralized.runner import run_a_scenario, to_agent_objects
from torch_geometric.data import Data

from multi_optim.state import ScenarioState

logger = logging.getLogger(__name__)


class Eval(object):
    """This class hold the information to evaluate the roadmap and policy
    during its run."""

    def __init__(self, roadmap: nx.Graph, map_img: MAP_IMG, n_agents: int,
                 n_eval: int, iterator_type: IteratorType, radius: float) -> None:
        torch.manual_seed(0)
        np.random.seed(0)
        self.rng = Random(0)
        self.first_roadmap = roadmap
        self.map_img = map_img
        self.n_agents = n_agents
        self.n_eval = n_eval
        self.iterator_type = iterator_type
        self.radius = radius
        pos = nx.get_node_attributes(self.first_roadmap, POS)
        pos_np = np.array([pos[n] for n in self.first_roadmap.nodes])
        flann = FLANN(random_seed=0)
        flann.build_index(np.array(pos_np, dtype=np.float32),
                          random_seed=0)
        self.untrained_policy = EdgePolicyModel()

        self.starts_corrds_s = []  # type: List[List[Tuple[float, float]]]
        self.goals_corrds_s = []  # type: List[List[Tuple[float, float]]]
        self.starts_s = []  # type: List[List[C]]
        self.goals_s = []  # type: List[List[C]]
        self.res_optimal_policy = []  # type: List[SCENARIO_RESULT]
        self.agents_s = []  # type: List[List[Agent]]
        self.states = []  # type: List[ScenarioState]
        self.eval_set_accuracy = []  # type: List[Tuple[Data, BFS_TYPE]]

        for i_e in range(self.n_eval):
            solvable_and_interesting = False
            while not solvable_and_interesting:
                unique = False
                starts = None  # type: Optional[List[C]]
                goals = None  # type: Optional[List[C]]
                starts_goals_coord: Optional[np.ndarray] = None
                while not unique:
                    starts_goals_coord = sample_points(
                        n_agents * 2, map_img, self.rng
                    ).detach().numpy()
                    nearest, _ = flann.nn_index(
                        starts_goals_coord,
                        1,
                        random_seed=0)
                    starts = nearest[:n_agents].tolist()
                    goals = nearest[n_agents:].tolist()
                    assert starts is not None
                    assert goals is not None
                    unique = len(set(starts)) == len(starts) and \
                        len(set(goals)) == len(goals)
                assert starts is not None
                assert goals is not None
                # make agents (only possible if single-agent sovable)
                agents = to_agent_objects(
                    self.first_roadmap, starts, goals,
                    policy=PolicyType.OPTIMAL, radius=self.radius,
                    rng=self.rng)
                if agents is None:
                    continue
                # genereally solvable?
                res_cbsr = plan_cbsr(self.first_roadmap, starts, goals,
                                     self.radius, DEFAULT_TIMEOUT_S,
                                     skip_cache=False)
                if res_cbsr == INVALID:
                    continue
                # solvable by policy
                res = run_a_scenario(self.first_roadmap,
                                     agents, False, self.iterator_type)
                if not res[IDX_SUCCESS]:
                    continue
                # maybe we can also make a state:
                state = ScenarioState(
                    self.first_roadmap, starts, goals,
                    self.untrained_policy, self.radius)
                state.run()
                if state.finished:
                    continue
                # get state and observation:
                self.states.append(state)
                observation = state.observe()
                assert observation is not None
                i_a = self.rng.sample(
                    list(observation.keys()), 1)[0]
                data, big_from_small = observation[i_a]
                self.eval_set_accuracy.append(
                    (data, big_from_small))
                # save meta data:
                assert starts_goals_coord is not None
                self.starts_corrds_s.append(
                    starts_goals_coord[:n_agents].tolist())
                self.goals_corrds_s.append(
                    starts_goals_coord[n_agents:].tolist())
                self.starts_s.append(starts)
                self.goals_s.append(goals)
                self.res_optimal_policy.append(res)
                self.agents_s.append(agents)
                solvable_and_interesting = True

    def evaluate_policy(self, model: EdgePolicyModel
                        ) -> Tuple[float, float, float]:
        """
        Evaluate the policy on the first roadmap.

        :param model: The model to evaluate.
        :return: The average regret, success and accuracy.
        """
        model.eval()
        success_s = np.zeros(self.n_eval)
        regret_s = np.zeros(self.n_eval)
        for i_e in range(self.n_eval):
            # prepare agents
            for i_a, a in enumerate(self.agents_s[i_e]):
                a.policy = LearnedPolicy(a, model)
                a.back_to_the_start()
            res = run_a_scenario(self.first_roadmap,
                                 self.agents_s[i_e], False, self.iterator_type)
            success_s[i_e] = res[IDX_SUCCESS]
            regret_s[i_e] = res[IDX_AVERAGE_LENGTH] - \
                self.res_optimal_policy[i_e][IDX_AVERAGE_LENGTH]
        accuracy = model.accuracy(self.eval_set_accuracy)
        return np.mean(regret_s), np.mean(success_s), accuracy

    def evaluate_roadmap(self, graph: nx.Graph, flann: FLANN) -> float:
        """
        Evaluate the roadmap for lengths of optimal paths.

        :param graph: The roadmap to evaluate.
        :return: The average length of the optimal paths.
        """
        pos = nx.get_node_attributes(graph, POS)
        pos_t = torch.tensor([pos[n].tolist() for n in graph.nodes])
        path_lens = []
        for i_e in range(self.n_eval):
            starts, _ = flann.nn_index(
                np.array(self.starts_corrds_s[i_e], dtype=np.float32),
                1, random_seed=0)
            goals, _ = flann.nn_index(
                np.array(self.goals_corrds_s[i_e], dtype=np.float32),
                1, random_seed=0)
            paths = plan_cbsr(graph, starts, goals, self.radius,
                              DEFAULT_TIMEOUT_S, skip_cache=False)
            if paths == INVALID:
                logger.warning("No paths")
                continue
            assert paths is not None
            for i_a in range(self.n_agents):
                path = (
                    self.starts_corrds_s[i_e][i_a],
                    self.goals_corrds_s[i_e][i_a],
                    [n for n, _ in paths[i_a]])
                path_lens.append(
                    get_path_len(pos_t, path, training=False).item())
        return float(np.mean(path_lens))

    def evaluate_both(self, model: EdgePolicyModel,
                      graph: nx.Graph, flann: FLANN
                      ) -> Tuple[float, float, float]:
        """
        Evaluate both the policy and the roadmap.

        :param model: The model to evaluate.
        :param graph: The roadmap to evaluate.
        :return: The average regret and success.
        """
        model.eval()
        success_s = np.zeros(self.n_eval)
        regret_s = []
        lenght_s = []
        for i_e in range(self.n_eval):
            starts, _ = flann.nn_index(
                np.array(self.starts_corrds_s[i_e], dtype=np.float32),
                1, random_seed=0)
            goals, _ = flann.nn_index(
                np.array(self.goals_corrds_s[i_e], dtype=np.float32),
                1, random_seed=0)
            # run sim with optimal policy
            agents = to_agent_objects(
                self.first_roadmap, starts.tolist(), goals.tolist(),
                policy=PolicyType.OPTIMAL, radius=self.radius,
                rng=self.rng)
            if agents is None:
                continue
            try:
                res_optimal = run_a_scenario(
                    graph, agents, False, self.iterator_type)
            except RuntimeError:
                continue
            # run sim with learned policy
            for i_a, a in enumerate(agents):
                a.policy = LearnedPolicy(a, model)
                a.back_to_the_start()
            res_policy = run_a_scenario(
                graph, agents, False, self.iterator_type)
            success_s[i_e] = res_policy[IDX_SUCCESS]
            regret_s.append(res_policy[IDX_AVERAGE_LENGTH] -
                            res_optimal[IDX_AVERAGE_LENGTH])
            lenght_s.append(res_policy[IDX_AVERAGE_LENGTH])
        return np.mean(regret_s), np.mean(success_s), np.mean(lenght_s)

# def eval_full_scenario(
#     model, g: nx.Graph, n_agents, n_eval, rng
# ) -> Tuple[Optional[float], float, float]:
#     regret_s = []
#     success_s = []
#     length_s = []
#     model.eval()
#     for i_e in range(n_eval):
#         # logger.debug(f"Eval {i_e}")
#         starts = rng.sample(g.nodes(), n_agents)
#         goals = rng.sample(g.nodes(), n_agents)
#         failed_at_creation = False

#         res_policy = (0., 0., 0., 0., 0)
#         res_optim = (0., 0., 0., 0., 0)
#         for policy in [OptimalPolicy, LearnedPolicy]:
#             agents = []
#             for i_a in range(n_agents):
#                 a = Agent(g, starts[i_a], radius=self.radius)
#                 res = a.give_a_goal(goals[i_a])
#                 if not res:  # failed to find a path
#                     failed_at_creation = True
#                 a.policy = policy(a, model)
#                 agents.append(a)
#             if not failed_at_creation:
#                 if policy is LearnedPolicy:
#                     res_policy = run_a_scenario(
#                         env=g,
#                         agents=tuple(agents),
#                         plot=False,
#                         iterator=self.iterator,
#                         ignore_finished_agents=False)
#                 elif policy is OptimalPolicy:
#                     try:
#                         res_optim = run_a_scenario(
#                             env=g,
#                             agents=tuple(agents),
#                             plot=False,
#                             iterator=self.iterator,
#                             ignore_finished_agents=False)
#                     except Exception as e:
#                         logger.error(e)

#         success = (res_policy[IDX_SUCCESS] == 1 and
#                    res_optim[IDX_SUCCESS] == 1)
#         # logger.debug(f"success: {success}")
#         if success:
#             regret = res_policy[IDX_AVERAGE_LENGTH] - \
#                 res_optim[IDX_AVERAGE_LENGTH]

#             if regret < 0:
#                 logger.warning("Regret is negative")
#             #     DEBUG
#             #     torch.save(model.state_dict(), "debug.pt")
#             #     nx.write_gpickle(g, f"debug.gpickle")
#             #     print(f"Starts: {starts}")
#             #     print(f"Goals: {goals}")
#             #     raise Exception("Regret is negative")

#             regret_s.append(regret)
#             length_s.append(res_policy[IDX_AVERAGE_LENGTH])
#         success_s.append(res_policy[IDX_SUCCESS])
#     if len(regret_s) > 0:
#         return np.mean(regret_s), np.mean(success_s), np.mean(length_s)
#     else:
#         return None, np.mean(success_s), np.mean(length_s)
