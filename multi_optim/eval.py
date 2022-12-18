import logging
from builtins import float
from copy import deepcopy
from pprint import pprint
from random import Random
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
from pyflann import FLANN
from torch_geometric.data import Data
from tqdm import tqdm

from definitions import (DEFAULT_TIMEOUT_S, IDX_AVERAGE_LENGTH, IDX_SUCCESS,
                         INVALID, MAP_IMG, POS, C)
from multi_optim.state import ScenarioState
from planner.mapf_implementations.plan_cbs_roadmap import plan_cbsr
from planner.policylearn.edge_policy import BFS_TYPE, EdgePolicyModel
from roadmaps.var_odrm_torch.var_odrm_torch import get_path_len, sample_points
from sim.decentralized.iterators import IteratorType
from sim.decentralized.policy import LearnedPolicy, PolicyType
from sim.decentralized.runner import run_a_scenario, to_agent_objects

logger = logging.getLogger(__name__)

SUCCESS_STR = "success"
REGRET_STR = "regret"
ACCURACY_STR = "accuracy"
LENGTH_STR = "length"


class Eval(object):
    """This class hold the information to evaluate the roadmap and policy
    during its run."""

    def __init__(self, roadmap: nx.Graph, map_img: MAP_IMG,
                 n_agents_s: List[int], n_eval_per_n_agents: int,
                 iterator_type: IteratorType, radius: float,
                 seed: int = 0) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.rng = Random(seed)
        self.first_roadmap = deepcopy(roadmap)
        self.map_img = map_img
        self.n_agents_s = n_agents_s
        self.n_eval = n_eval_per_n_agents * len(n_agents_s)
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
        self.first_lengths = []  # type: List[Optional[float]]
        # per n_agents ...
        self.eval_set_accuracy: List[
            List[Tuple[Data, BFS_TYPE]]] = [list() for _ in n_agents_s]

        N_TRIES = 100

        # progress bar
        pb_t = tqdm(total=self.n_eval, desc="eval", unit="sc")

        # stats
        stats_prototype = {
            "failed_generating_agents": 0,
            "failed_cbsr": 0,
            "failed_policy": 0,
            "failed_state_construction": 0,
            "failed_state_observe": 0,
            "success": 0
        }
        stats = {}

        for i_na, n_agents in enumerate(self.n_agents_s):
            stats[n_agents] = {}
            for i_e in range(n_eval_per_n_agents):
                stats[n_agents][i_e] = deepcopy(stats_prototype)
                this_stats = stats[n_agents][i_e]
                solvable_and_interesting = False
                pb_t.update(1)
                remaining_tries = N_TRIES
                while not solvable_and_interesting:
                    remaining_tries -= 1
                    if remaining_tries == 0:
                        logger.error(f"Failed to generate a scenario for "
                                     f"{n_agents} agents after "
                                     f"{N_TRIES} tries."
                                     f"stats: {stats}")
                        raise RuntimeError("Failed to generate a scenario")
                    unique = False
                    starts: Optional[List[C]] = None
                    goals: Optional[List[C]] = None
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
                        this_stats["failed_generating_agents"] += 1
                        continue
                    # genereally solvable?
                    res_cbsr = plan_cbsr(self.first_roadmap, starts, goals,
                                         self.radius, DEFAULT_TIMEOUT_S,
                                         skip_cache=False,
                                         ignore_finished_agents=True)
                    if res_cbsr == INVALID:
                        this_stats["failed_cbsr"] += 1
                        continue
                    # solvable by policy
                    res = run_a_scenario(self.first_roadmap,
                                         agents, False, self.iterator_type)
                    if not res[IDX_SUCCESS]:
                        this_stats["failed_policy"] += 1
                        continue
                    # maybe we can also make a state:
                    state = ScenarioState(
                        self.first_roadmap, starts, goals,
                        self.untrained_policy, self.radius)
                    state.run()
                    if state.finished:
                        this_stats["failed_state_construction"] += 1
                        continue
                    # get state and observation:
                    try:
                        observation = state.observe()
                    except RuntimeError:
                        # if no path was found, we can't use this state
                        this_stats["failed_state_observe"] += 1
                        continue
                    assert observation is not None
                    self.states.append(state)
                    i_a = self.rng.sample(
                        list(observation.keys()), 1)[0]
                    data, big_from_small = observation[i_a]
                    self.eval_set_accuracy[i_na].append(
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
                    self.first_lengths.append(None)
                    solvable_and_interesting = True
                    this_stats["success"] += 1
        pb_t.close()
        pprint(stats)

    def evaluate_policy(self, model: EdgePolicyModel
                        ) -> Dict[str, float]:
        """
        Evaluate the policy on the first roadmap.

        :param model: The model to evaluate.
        :return: The average regret, success and accuracy.
        """
        model.eval()
        results: Dict[int, Dict[str, List[float]]] = {}
        for i_na, n_agents in enumerate(self.n_agents_s):
            results[n_agents] = {}
            results[n_agents][SUCCESS_STR] = []
            results[n_agents][REGRET_STR] = []
            results[n_agents][ACCURACY_STR] = [model.accuracy(
                self.eval_set_accuracy[i_na])]
        for i_e in range(self.n_eval):
            # prepare agents
            n_agents = len(self.agents_s[i_e])
            for a in self.agents_s[i_e]:
                a.policy = LearnedPolicy(a, model)
                a.back_to_the_start()
            this_res = run_a_scenario(
                self.first_roadmap,
                self.agents_s[i_e],
                False,
                self.iterator_type)
            assert n_agents in results.keys()
            results[n_agents][SUCCESS_STR].append(
                this_res[IDX_SUCCESS])
            assert self.res_optimal_policy[i_e][IDX_SUCCESS]
            if this_res[IDX_SUCCESS]:
                results[n_agents][REGRET_STR].append(
                    this_res[IDX_AVERAGE_LENGTH] -
                    self.res_optimal_policy[i_e][IDX_AVERAGE_LENGTH])
        return_results: Dict[str, float] = {}
        for n_agents, measures in results.items():
            for measure_name, measure_values in measures.items():
                if len(measure_values) > 0:
                    return_results[
                        f"{measure_name}_{n_agents}"
                    ] = float(np.mean(measure_values))
        return return_results

    def evaluate_roadmap(self, graph: nx.Graph, flann: FLANN) -> float:
        """
        Evaluate the roadmap for lengths of optimal paths.

        :param graph: The roadmap to evaluate.
        :return: The average length of the optimal paths.
        """
        pos = nx.get_node_attributes(graph, POS)
        pos_t = torch.tensor([pos[n].tolist() for n in graph.nodes])
        all_e_lengths = []
        for i_e in range(self.n_eval):
            n_agents = len(self.starts_s[i_e])
            this_e_lens = []
            starts, _ = flann.nn_index(
                np.array(self.starts_corrds_s[i_e], dtype=np.float32),
                1, random_seed=0)
            goals, _ = flann.nn_index(
                np.array(self.goals_corrds_s[i_e], dtype=np.float32),
                1, random_seed=0)
            paths = plan_cbsr(graph, list(starts), list(goals), self.radius,
                              DEFAULT_TIMEOUT_S, skip_cache=False,
                              ignore_finished_agents=True)
            if paths == INVALID:
                logger.warning("No paths")
                continue
            assert paths is not None
            for i_a in range(n_agents):
                path = (
                    self.starts_corrds_s[i_e][i_a],
                    self.goals_corrds_s[i_e][i_a],
                    [n for n, _ in paths[i_a]])
                this_e_lens.append(get_path_len(
                    pos_t, path, training=False).item())
            this_e_len = float(np.mean(this_e_lens))
            if self.first_lengths[i_e] is None:
                self.first_lengths[i_e] = this_e_len
                all_e_lengths.append(1.)
            else:
                first_len = self.first_lengths[i_e]
                assert first_len is not None
                all_e_lengths.append(this_e_len / first_len)
        return float(np.mean(all_e_lengths))

    def evaluate_both(self, model: EdgePolicyModel,
                      graph: nx.Graph, flann: FLANN
                      ) -> Dict[str, float]:
        """
        Evaluate both the policy and the roadmap.

        :param model: The model to evaluate.
        :param graph: The roadmap to evaluate.
        :return: The average regret and success.
        """
        model.eval()
        results: Dict[int, Dict[str, List[float]]] = {}
        for i_na, n_agents in enumerate(self.n_agents_s):
            results[n_agents] = {}
            results[n_agents][SUCCESS_STR] = []
            results[n_agents][REGRET_STR] = []
            results[n_agents][LENGTH_STR] = []
        for i_e in range(self.n_eval):
            n_agents = len(self.starts_s[i_e])
            starts, _ = flann.nn_index(
                np.array(self.starts_corrds_s[i_e], dtype=np.float32),
                1, random_seed=0)
            goals, _ = flann.nn_index(
                np.array(self.goals_corrds_s[i_e], dtype=np.float32),
                1, random_seed=0)
            # run sim with optimal policy
            agents = to_agent_objects(
                graph, starts.tolist(), goals.tolist(),
                policy=PolicyType.OPTIMAL, radius=self.radius,
                rng=self.rng)
            if agents is None:
                continue
            try:
                res_optimal = run_a_scenario(
                    graph, agents, False, self.iterator_type)
            except RuntimeError:
                res_optimal = (0, 0, 0, 0, 0)
            # run sim with learned policy
            for i_a, a in enumerate(agents):
                a.policy = LearnedPolicy(a, model)
                a.back_to_the_start()
            res_policy = run_a_scenario(
                graph, agents, False, self.iterator_type)

            # success
            if res_policy[IDX_SUCCESS]:
                results[n_agents][SUCCESS_STR].append(1.)
            else:
                results[n_agents][SUCCESS_STR].append(0.)

            if res_policy[IDX_SUCCESS] and res_optimal[IDX_SUCCESS]:
                # regret
                this_regret = (res_policy[IDX_AVERAGE_LENGTH] -
                               res_optimal[IDX_AVERAGE_LENGTH])
                results[n_agents][REGRET_STR].append(this_regret)

                # length
                if self.first_lengths[i_e] is None:
                    self.first_lengths[i_e] = res_policy[IDX_AVERAGE_LENGTH]
                    results[n_agents][LENGTH_STR].append(1.)
                else:
                    first_len = self.first_lengths[i_e]
                    assert first_len is not None
                    this_len = (res_policy[IDX_AVERAGE_LENGTH] /
                                first_len)
                    results[n_agents][LENGTH_STR].append(this_len)

        return_results: Dict[str, float] = {}
        for n_agents, measures in results.items():
            for measure_name, measure_values in measures.items():
                if len(measure_values) > 0:
                    return_results[f"{measure_name}_{n_agents}"
                                   ] = float(np.mean(measure_values))
        return return_results
