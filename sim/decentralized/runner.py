#!/usr/bin/env python3

import itertools
import logging
import random
from typing import *

import numpy as np
from definitions import BLOCKED_NODES_TYPE, INVALID, SCENARIO_RESULT
from sim.decentralized.agent import Agent, env_to_nx
from sim.decentralized.iterators import (IteratorType, SimIterationException,
                                         get_iterator_fun)
from sim.decentralized.policy import PolicyCalledException, PolicyType
from sim.decentralized.visualization import *
from tools import ProgressBar

logging.basicConfig()
logger = logging.getLogger(__name__)
TIME_LIMIT = 100


def initialize_environment_random_fill(size: int, fill: float, rng: random.Random = random.Random()
                                       ) -> np.ndarray:
    """Make a square map with edge length `size` and `fill` (0..1) obstacle
    ratio.
    :param size: side length of the square (in pixels)
    :param fill: percentage of map to fill
    :return: the environment
    """
    environment = np.zeros([size, size], dtype=np.int8)
    n_to_fill = int(fill * size ** 2)
    to_fill = rng.sample(
        list(itertools.product(range(size), repeat=2)), k=n_to_fill)
    for cell in to_fill:
        environment[cell] = 1
    return environment


def initialize_new_agent(
        env: np.ndarray, agents: List[Agent], policy: PolicyType,
        tight_placement: bool = False,
        rng: random.Random = random.Random()) -> Optional[Agent]:
    """Place new agent in the environment, where no obstacle or other agent
    is.

    :param tight_placement: if false, start and goal places are sampled from
    the same distribution
    :raises AssertionError if no space is left
    :return: the agent
    """
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
        pos: int = rng.choice(list(np.transpose(no_obstacle_nor_agent)))
        a = Agent(env, pos, policy, rng=rng)  # we have the agent

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
        pos = rng.choice(np.transpose(
            no_obstacle_nor_agent_or_goal))  # type: ignore
        a = Agent(env, pos, policy, rng=rng)  # we have the agent

        # now finding the goal ...
        if a.has_gridmap:
            assert isinstance(pos, Iterable)
            env_with_agents_and_goals[tuple(pos)] = 1
        elif a.has_roadmap:
            env_with_agents_and_goals[pos] = 1
        no_obstacle_nor_agent_or_goal = np.where(
            env_with_agents_and_goals == 0)
        assert len(no_obstacle_nor_agent_or_goal[0]
                   ) > 0, "Possible poses should be left"
        goals = np.transpose(no_obstacle_nor_agent_or_goal)

    goals = list(goals)
    rng.shuffle(goals)
    for g in goals:
        if a.give_a_goal(g):  # success
            return a
    return None


def to_agent_objects(env, starts, goals, policy=PolicyType.RANDOM,
                     rng: random.Random = random.Random()):
    n_agents = starts.shape[0]
    agents = []
    env_nx = env_to_nx(env)
    for i_a in range(n_agents):
        a = Agent(env, starts[i_a], policy=policy, rng=rng, env_nx=env_nx)
        if not a.give_a_goal(goals[i_a]):
            return INVALID
        agents.append(a)
    return agents


def initialize_agents(
        env: np.ndarray, n_agents: int, policy: PolicyType,
        tight_placement: bool = False, rng: random.Random = random.Random()
) -> Optional[Tuple[Agent, ...]]:
    """Initialize `n_agents` many agents in unique, free spaces of
    `environment`, (not colliding with each other). Returns None if one agent
    could not find a path to its goal."""
    agents: List[Agent] = []  # starting with a list for easy inserting
    for _ in range(n_agents):
        agent = initialize_new_agent(
            env, agents, policy, tight_placement, rng)
        if agent is None:
            return None
        agents.append(agent)
    return tuple(agents)  # returning tuple because it can be immutable now


def is_environment_well_formed(agents: Tuple[Agent]) -> bool:
    """Check if the environment is well formed according to Cap2015"""
    for a in agents:
        if isinstance(a.env, np.ndarray):
            a.env = a.env.copy()
        blocks: BLOCKED_NODES_TYPE = set()
        for other_a in [ia for ia in agents if ia != a]:
            assert other_a.goal is not None, "Other agent should have a goal"
            blocks.add(other_a.pos)
            blocks.add(other_a.goal)
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


def will_scenario_collide_and_get_paths(env, starts, goals, ignore_finished_agents):
    """checks if for a given set of starts and goals the agents travelling
    between may collide on the given env."""
    return will_agents_collide(to_agent_objects(env, starts, goals), ignore_finished_agents)


def will_scenario_collide(env, starts, goals, ignore_finished_agents) -> Optional[bool]:
    """checks if for a given set of starts and goals the agents travelling
    between may collide on the given env."""
    if ignore_finished_agents == False:
        raise NotImplementedError()
    env_nx = env_to_nx(env)
    n_agents = starts.shape[0]
    seen = set()
    for i_a in range(n_agents):
        a = Agent(env, starts[i_a], env_nx=env_nx)
        if not a.give_a_goal(goals[i_a]):
            return None
        assert a.path is not None, "Agent should have a path"
        for pos in a.path:
            if pos in seen:
                return True
            seen.add(pos)
    return False


def will_agents_collide(agents, ignore_finished_agents):
    """Checks agents in list if they will collide."""
    if ignore_finished_agents == False:
        raise NotImplementedError()
    do_collide = False
    seen = set()
    agent_paths = []
    for i_a, a in enumerate(agents):
        agent_paths.append(a.path)
        for pos in a.path:
            if not do_collide:  # only need to do this if no collision was found
                if tuple(pos) in seen:
                    do_collide = True
                seen.add(tuple(pos))
    return do_collide, agent_paths


def sample_and_run_a_scenario(size, n_agents, policy, plot, rng: random.Random, iterator
                              ) -> SCENARIO_RESULT:
    env = initialize_environment_random_fill(size, .1, rng)
    agents = initialize_agents(env, n_agents, policy, rng=rng)
    if agents is None:
        logger.warning("Could not initialize agents")
        return (0, 0, 0, 0, 0)
    res = run_a_scenario(env, agents, plot, iterator)
    assert not has_exception(res)
    return res


def run_a_scenario(env, agents, plot,
                   iterator: IteratorType = IteratorType.WAITING,
                   pause_on: Optional[Exception] = None,
                   ignore_finished_agents=True,
                   print_progress=False,
                   time_limit=TIME_LIMIT,
                   ) -> SCENARIO_RESULT:
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
            time_slice, space_slice = get_iterator_fun(
                iterator)(agents, ignore_finished_agents)
            time_progress += time_slice
            space_progress += space_slice
            if any(time_progress > time_limit):
                raise SimIterationException("timeout")
            finished = sum(map(lambda a: a.is_at_goal(), agents))
            logger.debug(
                f"t:{max(time_progress)} finished: {finished}/{n_agents}")
        successful = 1
        logger.debug('success')
    except Exception as e:  # pragma: no cover
        logger.debug(f'Exception: {e.__class__.__name__}, {e}')
        if isinstance(e, SimIterationException):
            pass  # logger.warning(e)
        elif pause_on is not None and isinstance(e, pause_on):  # type: ignore
            assert isinstance(e, PolicyCalledException)
            return check_time_evaluation(
                time_progress,
                space_progress) + (e, )
        else:
            raise e
    return check_time_evaluation(
        time_progress,
        space_progress) + (successful, )


def has_exception(res: SCENARIO_RESULT):
    return isinstance(res[-1], PolicyCalledException)


def evaluate_policies(size=10, n_agents=10, runs=100, plot_eval=True):
    """run the simulation with some policies"""
    rng = random.Random(0)
    evaluations = {}
    evaluation_names = [
        "average_time",
        "max_time",
        "average_length",
        "max_length",
        "successful"
    ]
    # policies = PolicyType
    policies = [PolicyType.CLOSEST,
                PolicyType.FILL,
                PolicyType.RANDOM,
                PolicyType.LEARNED,
                PolicyType.INVERSE_LEARNED]
    pb = ProgressBar("evaluate_policies", len(policies)*runs, 5)

    for policy in policies:
        logger.info(f"policy: {policy.name}")
        evaluation_per_policy = np.empty([len(evaluation_names), 0])
        for _ in range(runs):
            results = sample_and_run_a_scenario(
                size, n_agents, policy, False, rng, IteratorType.BLOCKING3)
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
