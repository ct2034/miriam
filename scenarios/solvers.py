import random

import numpy as np
from definitions import DEFAULT_TIMEOUT_S, INVALID
from planner.mapf_implementations.plan_ecbs import plan_in_gridmap
from planner.mapf_implementations.plan_cbs_roadmap import plan_cbsr
from planner.matteoantoniazzi_mapf.plan import icts_plan, paths_from_info
from sim.decentralized.iterators import IteratorType
from sim.decentralized.runner import run_a_scenario, to_agent_objects

from scenarios import storage
from scenarios.storage import ResultType

# ecbs ########################################################################

SCHEDULE = 'schedule'
AGENT = 'agent'


def cached_ecbs(env, starts, goals,
                timeout=DEFAULT_TIMEOUT_S, suboptimality=1.0,
                disappear_at_goal=True, skip_cache=False):
    if skip_cache:
        data = ecbs(env, starts, goals, timeout=timeout)
    else:
        scenario = (env, starts, goals)
        solver_params = {
            "timeout": timeout,
            "suboptimality": suboptimality,
            "disappear_at_goal": disappear_at_goal
        }
        if storage.has_result(scenario, ResultType.ECBS_DATA,
                              solver_params):
            data = storage.get_result(
                scenario, ResultType.ECBS_DATA, solver_params)
        else:
            data = ecbs(env, starts, goals, timeout=timeout)
            storage.save_result(scenario, ResultType.ECBS_DATA,
                                solver_params, data)
    return data


def ecbs(env, starts, goals, timeout=DEFAULT_TIMEOUT_S, suboptimality=1.0,
         disappear_at_goal=True, return_paths=False):
    """Plan scenario using ecbs returning results data"""
    try:
        data = plan_in_gridmap(env, list(starts), list(
            goals), suboptimality=suboptimality, timeout=timeout,
            disappear_at_goal=disappear_at_goal)
    except KeyError:  # happens when start or goal is not in map
        return INVALID
    if data is None or data is INVALID:
        return INVALID
    n_agents = starts.shape[0]
    if SCHEDULE not in data.keys():
        return INVALID
    schedule = data[SCHEDULE]
    if schedule is None:
        return INVALID
    if n_agents != len(schedule.keys()):  # not plans for all agents
        return INVALID
    if return_paths:
        return _ecbs_data_to_paths(data)
    else:
        return data


def _ecbs_data_to_paths(data):
    schedule = data[SCHEDULE]
    n_agents = len(schedule.keys())
    paths = []
    for i_a in range(n_agents):
        key = AGENT + str(i_a)
        assert key in schedule.keys()
        one_path = list(map(
            lambda d: [d['x'], d['y'], d['t']],
            schedule[key]
        ))
        paths.append(np.array(one_path))
    return paths


# cbs roadmaps ################################################################


def cached_cbsr(env, starts, goals,
                timeout=DEFAULT_TIMEOUT_S,
                radius: float = 0.1,
                skip_cache=False):
    if skip_cache:
        paths = plan_cbsr(env, starts, goals, radius=radius,
                          timeout=timeout, skip_cache=skip_cache)
    else:
        scenario = (env, starts, goals)
        solver_params = {
            "radius": radius,
            "timeout": timeout
        }
        if storage.has_result(scenario, ResultType.CBSR_PATHS,
                              solver_params):
            paths = storage.get_result(
                scenario, ResultType.CBSR_PATHS, solver_params)
        else:
            paths = plan_cbsr(env, starts, goals, radius=radius,
                              timeout=timeout, skip_cache=skip_cache)
            storage.save_result(scenario, ResultType.CBSR_PATHS,
                                solver_params, paths)
    return paths


# icts ########################################################################


def cached_icts(env, starts, goals,
                timeout=DEFAULT_TIMEOUT_S, skip_cache=False):
    scenario = (env, starts, goals)
    if skip_cache:
        info = icts(env, starts, goals, timeout)
    else:
        if storage.has_result(scenario, ResultType.ICTS_INFO, {}):
            info = storage.get_result(
                scenario, ResultType.ICTS_INFO, {})
        else:
            info = icts(env, starts, goals, timeout)
            storage.save_result(scenario, ResultType.ICTS_INFO, {}, info)
    return info


def icts(env, starts, goals, timeout=DEFAULT_TIMEOUT_S, return_paths=False):
    info = icts_plan(env, starts, goals, timeout)
    if return_paths:
        return paths_from_info(info)
    else:
        return info

# decentralized ###############################################################


def indep(env, starts, goals):
    agents = to_agent_objects(env, starts, goals)
    if agents is None:
        return INVALID
    paths = [a.path for a in agents]
    return paths


def cached_decentralized(env, starts, goals, policy,
                         ignore_finished_agents, skip_cache=False):
    solver_params = {
        "policy": policy.name,
        "ignore_finished_agents": ignore_finished_agents
    }
    scenario = (env, starts, goals)
    if skip_cache:
        metrics = decentralized(
            env, starts, goals, policy, ignore_finished_agents)
    else:
        if storage.has_result(scenario, ResultType.DECEN, solver_params):
            metrics = storage.get_result(
                scenario, ResultType.DECEN, solver_params)
        else:
            metrics = decentralized(
                env, starts, goals, policy, ignore_finished_agents)
            storage.save_result(scenario, ResultType.DECEN,
                                solver_params, metrics)
    return metrics


def decentralized(env, starts, goals,
                  policy, ignore_finished_agents):
    agents = to_agent_objects(
        env, starts, goals, policy, rng=random.Random(0))
    if agents is None:
        return INVALID
    assert agents is not None
    return run_a_scenario(
        env, agents, plot=False, iterator=IteratorType.LOOKAHEAD1,
        ignore_finished_agents=ignore_finished_agents)
