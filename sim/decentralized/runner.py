#!/usr/bin/env python3

import itertools
import logging
import random
from typing import *

import numpy as np
from sim.decentralized import iterators
from sim.decentralized.agent import Agent
from sim.decentralized.iterators import (IteratorType, SimIterationException,
                                         get_iterator_fun)
from sim.decentralized.policy import PolicyType
from sim.decentralized.visualization import *
from tools import ProgressBar

logging.basicConfig()
logger = logging.getLogger(__name__)


def initialize_environment(size: int, fill: float, seed: float = random.random()
                           ) -> np.ndarray:
    """Make a square map with edge length `size` and `fill` (0..1) obstacle
    ratio.
    :param size: side length of the square (in pixels)
    :param fill: percentage of map to fill
    :return: the environment
    """
    random.seed(seed)
    environment = np.zeros([size, size], dtype=np.int8)
    n_to_fill = int(fill * size ** 2)
    to_fill = random.sample(
        list(itertools.product(range(size), repeat=2)), k=n_to_fill)
    for cell in to_fill:
        environment[cell] = 1
    return environment


def initialize_new_agent(
        env: np.ndarray, agents: List[Agent], policy: PolicyType,
        tight_placement: bool = False, seed: float = random.random()) -> Optional[Agent]:
    """Place new agent in the environment, where no obstacle or other agent
    is.

    :param tight_placement: if false, start and goal places are sampled from
    the same distribution
    :raises AssertionError if no space is left
    :return: the agent
    """
    random.seed(seed)
    np.random.seed(int(seed*1000))
    if tight_placement:  # starts can be at other goals
        env_with_agents = env.copy()
        env_with_goals = env.copy()
        for a in agents:
            env_with_agents[tuple(a.pos)] = 1
            assert a.goal is not None, "Agent should have a goal"
            env_with_goals[tuple(a.goal)] = 1
        no_obstacle_nor_agent = np.where(env_with_agents == 0)
        no_obstacle_nor_goal = np.where(env_with_goals == 0)
        assert len(no_obstacle_nor_agent[0]
                   ) > 0, "Possible poses should be left"
        assert len(no_obstacle_nor_goal[0]
                   ) > 0, "Possible poses should be left"
        pos = random.choice(np.transpose(no_obstacle_nor_agent))
        a = Agent(env, pos, policy)  # we have the agent

        # now finding the goal ...
        goals = np.transpose(no_obstacle_nor_goal)
    else:  # no tight placement: sample starts and goals from same distribution
        env_with_agents_and_goals = env.copy()
        for a in agents:
            env_with_agents_and_goals[tuple(a.pos)] = 1
            assert a.goal is not None, "Agent should have a goal"
            env_with_agents_and_goals[tuple(a.goal)] = 1
        no_obstacle_nor_agent_or_goal = np.where(
            env_with_agents_and_goals == 0)
        assert len(no_obstacle_nor_agent_or_goal[0]
                   ) > 0, "Possible poses should be left"
        pos = random.choice(np.transpose(no_obstacle_nor_agent_or_goal))
        a = Agent(env, pos, policy)  # we have the agent

        # now finding the goal ...
        env_with_agents_and_goals[tuple(pos)] = 1
        no_obstacle_nor_agent_or_goal = np.where(
            env_with_agents_and_goals == 0)
        assert len(no_obstacle_nor_agent_or_goal[0]
                   ) > 0, "Possible poses should be left"
        goals = np.transpose(no_obstacle_nor_agent_or_goal)

    np.random.shuffle(goals)
    for g in goals:
        if a.give_a_goal(g):  # success
            return a
    return None


def initialize_agents(
        env: np.ndarray, n_agents: int, policy: PolicyType,
        tight_placement: bool = False, seed: float = random.random()
) -> Optional[Tuple[Agent, ...]]:
    """Initialize `n_agents` many agents in unique, free spaces of
    `environment`, (not colliding with each other). Returns None if one agent
    could not find a path to its goal."""
    random.seed(seed)
    np.random.seed(int(seed*1000))
    agents: List[Agent] = []  # starting with a list for easy inserting
    for _ in range(n_agents):
        agent_seed = random.random()
        agent = initialize_new_agent(
            env, agents, policy, tight_placement, agent_seed)
        if agent is None:
            return None
        agents.append(agent)
    return tuple(agents)  # returning tuple because it can be immutable now


def is_environment_well_formed(agents: Tuple[Agent]) -> bool:
    """Check if the environment is well formed according to Cap2015"""
    for a in agents:
        blocks: List[Tuple[Any, ...]] = []
        for other_a in [ia for ia in agents if ia != a]:
            blocks.append(tuple(other_a.pos))
            assert other_a.goal is not None, "Other agent should have a goal"
            blocks.append(tuple(other_a.goal))
        if not a.is_there_path_with_node_blocks(blocks):
            return False
    return True


def are_all_agents_at_their_goals(agents: List[Agent]) -> bool:
    """Returns true iff all agents are at their respective goals."""
    return all(map(lambda a: a.is_at_goal(), agents))


def check_time_evaluation(time_progress, space_progress
                          ) -> Tuple[float, float, float, float]:
    average_time = sum(time_progress) / len(time_progress)
    max_time = max(time_progress)
    average_length = sum(space_progress) / len(space_progress)
    max_length = max(space_progress)
    return average_time, max_time, average_length, max_length


def sample_and_run_a_scenario(size, n_agents, policy, plot, seed, iterator
                              ) -> Tuple[float, float, float, float, int]:
    env = initialize_environment(size, .1, seed)
    agents = initialize_agents(env, n_agents, policy, seed=seed)
    if agents is None:
        logger.warning("Could not initialize agents")
        return (0, 0, 0, 0, 0)
    return run_a_scenario(env, agents, plot, iterator)


def run_a_scenario(env, agents, plot, iterator: IteratorType = IteratorType.WAITING
                   ) -> Tuple[float, float, float, float, int]:
    n_agents = len(agents)
    # evaluation parameters
    time_progress = np.zeros([n_agents], dtype=float)
    space_progress = np.zeros([n_agents], dtype=float)
    # iterate
    successful: int = 0
    try:
        while not are_all_agents_at_their_goals(agents):
            if plot:  # pragma: no cover
                plot_env_agents(env, agents)
            time_slice, space_slice = get_iterator_fun(iterator)(agents)
            time_progress += time_slice
            space_progress += space_slice
        successful = 1
    except SimIterationException as e:  # pragma: no cover
        logger.warning(e)

    return check_time_evaluation(
        time_progress,
        space_progress) + (successful, )


def evaluate_policies(size=10, n_agents=10, runs=100, plot_eval=True):
    """run the simulation with all policies"""
    evaluations = {}
    evaluation_names = [
        "average_time",
        "max_time",
        "average_length",
        "max_length",
        "successful"
    ]
    policies = PolicyType
    # policies = [PolicyType.RANDOM, PolicyType.LEARNED, PolicyType.INV_LEARNED]
    pb = ProgressBar("evaluate_policies", len(policies)*runs, 5)

    for policy in policies:
        logger.info(f"policy: {policy.name}")
        evaluation_per_policy = np.empty([len(evaluation_names), 0])
        for i_r in range(runs):
            random.seed(i_r)
            np.random.seed(int(i_r*1000))
            results = sample_and_run_a_scenario(
                size, n_agents, policy, False, i_r, IteratorType.BLOCKING3)
            evaluation_per_policy = np.append(
                evaluation_per_policy, np.transpose([results]), axis=1)
            pb.progress()
        evaluations[policy] = evaluation_per_policy
    pb.end()

    if plot_eval:  # pragma: no cover
        plot_evaluations(evaluations, evaluation_names, PolicyType.RANDOM)
    return (evaluations, evaluation_names)


if __name__ == "__main__":  # pragma: no cover
    logging.getLogger("sim.decentralized.agent").setLevel(logging.ERROR)
    logging.getLogger("sim.decentralized.policy").setLevel(logging.ERROR)
    logging.getLogger("__main__").setLevel(logging.ERROR)
    logging.getLogger("root").setLevel(logging.ERROR)
    evaluate_policies(8, 8, 100)
