import itertools
import logging
import random
from typing import Iterable, List, Optional, Tuple, Type, Union

import numpy as np
from definitions import BLOCKED_NODES_TYPE, INVALID, POS, SCENARIO_RESULT
from scenarios.types import POTENTIAL_ENV_TYPE, is_gridmap, is_roadmap
from sim.decentralized.agent import Agent
from sim.decentralized.iterators import (IteratorType, SimIterationException,
                                         get_iterator_fun)
from sim.decentralized.policy import PolicyCalledException, PolicyType
from sim.decentralized.visualization import *
from tools import ProgressBar

logging.basicConfig()
logger = logging.getLogger(__name__)
TIME_LIMIT = 100


def initialize_environment_random_fill(size: int, fill: float, rng: random.Random = random.Random(0)
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


def get_int_tuple_pos(a: Agent, node: int) -> Tuple[int, int]:
    assert a.has_gridmap
    pos_coord_f = a.env.nodes()[node][POS]
    pos_coord = (int(pos_coord_f[0]), int(pos_coord_f[1]))
    return pos_coord


def initialize_new_agent(
        gridmap: np.ndarray, agents: List[Agent], policy: PolicyType,
        tight_placement: bool = False,
        rng: random.Random = random.Random(0)) -> Optional[Agent]:
    """Place new agent in the environment, where no obstacle or other agent
    is.

    :param tight_placement: if false, start and goal places are sampled from
    the same distribution
    :raises AssertionError if no space is left
    :return: the agent
    """
    if tight_placement:  # starts can be at other goals
        gridmap_with_agents = gridmap.copy()
        env_with_goals = gridmap.copy()
        for a in agents:
            pos_coord = get_int_tuple_pos(a, a.pos)
            gridmap_with_agents[pos_coord] = 1
            assert a.goal is not None, "Agent should have a goal"
            goal_coord = get_int_tuple_pos(a, a.goal)
            env_with_goals[goal_coord] = 1
        no_obstacle_nor_agent = np.where(gridmap_with_agents == 0)
        no_obstacle_nor_goal = np.where(env_with_goals == 0)
        assert len(no_obstacle_nor_agent[0]
                   ) > 0, "Possible poses should be left"
        assert len(no_obstacle_nor_goal[0]
                   ) > 0, "Possible poses should be left"
        pos: int = rng.choice(list(np.transpose(no_obstacle_nor_agent)))
        a = Agent(gridmap, pos, policy, rng=rng)  # we have the agent

        # now finding the goal ...
        goals = np.transpose(no_obstacle_nor_goal)
    else:  # no tight placement: sample starts and goals from same distribution
        gridmap_with_agents_and_goals = gridmap.copy()
        for a in agents:
            pos_coord = get_int_tuple_pos(a, a.pos)
            gridmap_with_agents_and_goals[pos_coord] = 1
            assert a.goal is not None, "Agent should have a goal"
            goal_coord = get_int_tuple_pos(a, a.goal)
            gridmap_with_agents_and_goals[goal_coord] = 1
        no_obstacle_nor_agent_or_goal = np.where(
            gridmap_with_agents_and_goals == 0)
        assert len(no_obstacle_nor_agent_or_goal[0]
                   ) > 0, "Possible poses should be left"
        pos = rng.choice(np.transpose(
            no_obstacle_nor_agent_or_goal))  # type: ignore
        a = Agent(gridmap, pos, policy, rng=rng)  # we have the agent

        # now finding the goal ...
        if a.has_gridmap:
            assert isinstance(pos, Iterable)
            gridmap_with_agents_and_goals[tuple(pos)] = 1
        elif a.has_roadmap:
            gridmap_with_agents_and_goals[pos] = 1
        no_obstacle_nor_agent_or_goal = np.where(
            gridmap_with_agents_and_goals == 0)
        assert len(no_obstacle_nor_agent_or_goal[0]
                   ) > 0, "Possible poses should be left"
        goals = np.transpose(no_obstacle_nor_agent_or_goal)

    goals_tmp = list(goals)
    rng.shuffle(goals_tmp)
    goals = np.array(goals_tmp)
    for g in goals:
        if a.give_a_goal(g):  # success
            return a
    return None


def to_agent_objects(env, starts, goals, policy=PolicyType.RANDOM,
                     radius: Optional[float] = None,
                     rng: random.Random = random.Random(0)) -> Optional[List[Agent]]:
    n_agents = np.array(starts).shape[0]
    agents = []
    for i_a in range(n_agents):
        if is_gridmap(env):
            a = Agent(env, starts[i_a], policy=policy, rng=rng,
                      radius=radius)
        elif is_roadmap(env):
            a = Agent(env, int(starts[i_a]), policy=policy, rng=rng,
                      radius=radius)
        else:
            raise RuntimeError("Unknown environment type")
        if not a.give_a_goal(goals[i_a]):
            return None
        agents.append(a)
    return agents


def initialize_agents(
        env: np.ndarray, n_agents: int, policy: PolicyType,
        tight_placement: bool = False, rng: random.Random = random.Random(0)
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


def are_all_agents_at_their_goals(agents: Tuple[Agent, ...]) -> bool:
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
    n_agents = starts.shape[0]
    seen = set()
    for i_a in range(n_agents):
        a = Agent(env, starts[i_a])
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
                if pos in seen:
                    do_collide = True
                seen.add(pos)
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


def run_a_scenario(env: POTENTIAL_ENV_TYPE,
                   agents: Union[Tuple[Agent, ...], List[Agent]],
                   plot: bool,
                   iterator: IteratorType = IteratorType.LOOKAHEAD1,
                   pause_on: Optional[Type[Exception]] = None,
                   ignore_finished_agents: bool = True,
                   time_limit: int = TIME_LIMIT,
                   paths_out: Optional[List[List[int]]] = None,
                   ) -> SCENARIO_RESULT:
    n_agents = len(agents)
    if isinstance(agents, list):
        agents = tuple(agents)
    # evaluation parameters
    time_progress = np.zeros([n_agents], dtype=float)
    space_progress = np.zeros([n_agents], dtype=float)
    # output of paths
    if paths_out is None:
        # then the paths are used only internally
        paths_out = list(map(lambda _: list(), range(n_agents)))
    elif isinstance(paths_out, list) and len(paths_out) == 0:
        # initialized with an empty list
        for _ in range(n_agents):
            paths_out.append(list())
    else:
        assert isinstance(paths_out, list) and len(paths_out) == n_agents, \
            "paths_out should have been filled before"
    # oscillation parameters
    oscillation_count = 0
    max_oscillation_count = 1
    # iterate
    successful: int = 0
    try:
        # poses before moving
        while not are_all_agents_at_their_goals(agents):
            poses_before_moving = list(map(lambda a: a.pos, agents))
            if plot:  # pragma: no cover
                plot_env_agents(env, agents)  # type: ignore
            time_slice, space_slice = get_iterator_fun(
                iterator)(agents, ignore_finished_agents)
            time_progress += time_slice
            space_progress += space_slice
            if any(time_progress > time_limit):
                raise SimIterationException("timeout")
            finished = sum(map(lambda a: a.is_at_goal(), agents))
            logger.debug(
                f"t:{max(time_progress)} finished: {finished}/{n_agents}")
            t = len(paths_out[0]) - 1
            # taking care of the paths_out
            for i_a, p in enumerate(poses_before_moving):
                paths_out[i_a].append(p)
            if are_all_agents_at_their_goals(agents):
                for i_a, a in enumerate(agents):
                    paths_out[i_a].append(a.pos)
            # check same poses two timesteps ago
            if t >= 2:
                if all(map(
                        lambda i_a: paths_out[i_a][-3]  # type: ignore
                    == paths_out[i_a][-1],  # type: ignore
                        range(n_agents))):
                    oscillation_count += 1
            if oscillation_count > max_oscillation_count:
                raise SimIterationException("oscillation deadlock")
        successful = 1
        logger.debug('success')
        lengths = list(map(len, paths_out))
        assert all(map(lambda l: l == lengths[0], lengths))
    except Exception as e:  # pragma: no cover
        logger.debug(f'Exception: {e.__class__.__name__}, {e}')
        if isinstance(e, SimIterationException):
            pass  # logger.warning(e)
        elif pause_on is not None:
            if isinstance(e, pause_on):  # type: ignore
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
    policies = [PolicyType.RANDOM,
                PolicyType.LEARNED,
                PolicyType.OPTIMAL]
    pb = ProgressBar("evaluate_policies", len(policies)*runs, 5)

    for policy in policies:
        logger.info(f"policy: {policy.name}")
        evaluation_per_policy = np.empty([len(evaluation_names), 0])
        for _ in range(runs):
            results = sample_and_run_a_scenario(
                size, n_agents, policy, False, rng, IteratorType.LOOKAHEAD3)
            evaluation_per_policy = np.append(
                evaluation_per_policy, np.transpose(np.array([results])), axis=1)
            pb.progress()
        evaluations[policy] = evaluation_per_policy
    pb.end()

    if plot_eval:  # pragma: no cover
        plot_evaluations(evaluations, evaluation_names, PolicyType.OPTIMAL)
    return (evaluations, evaluation_names)


if __name__ == "__main__":  # pragma: no cover
    logging.getLogger("sim.decentralized.agent").setLevel(logging.INFO)
    logging.getLogger("sim.decentralized.policy").setLevel(logging.INFO)
    logging.getLogger("__main__").setLevel(logging.INFO)
    logging.getLogger("root").setLevel(logging.INFO)
    evaluate_policies(4, 2, 10)
