import numpy as np
from definitions import DEFAULT_TIMEOUT_S, INVALID
from planner.matteoantoniazzi_mapf.plan import icts_plan, paths_from_info
from planner.policylearn.libMultiRobotPlanning.plan_ecbs import plan_in_gridmap
from sim.decentralized.agent import Agent
from sim.decentralized.iterators import IteratorType
from sim.decentralized.policy import PolicyType
from sim.decentralized.runner import run_a_scenario

from scenarios import storage

SCHEDULE = 'schedule'
AGENT = 'agent'

# ecbs ########################################################################


def cached_ecbs(env, starts, goals,
                timeout=DEFAULT_TIMEOUT_S, skip_cache=False):
    if skip_cache:
        data = ecbs(env, starts, goals, timeout=timeout)
    else:
        scenario = (env, starts, goals)
        if storage.has_result(scenario, storage.ResultType.ECBS_DATA):
            data = storage.get_result(
                scenario, storage.ResultType.ECBS_DATA)
        else:
            data = ecbs(env, starts, goals, timeout=timeout)
            storage.save_result(scenario, storage.ResultType.ECBS_DATA, data)
    return data


def ecbs(env, starts, goals, timeout=DEFAULT_TIMEOUT_S, return_paths=False):
    """Plan scenario using ecbs returning results data"""
    try:
        data = plan_in_gridmap(env, list(starts), list(
            goals), suboptimality=1.0, timeout=timeout)  # OPTIMAL !
    except KeyError:  # happens when start or goal is not in map
        return INVALID
    if data is None:
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

# icts ########################################################################


def cached_icts(env, starts, goals,
                timeout=DEFAULT_TIMEOUT_S, skip_cache=False):
    scenario = (env, starts, goals)
    if skip_cache:
        info = icts(env, starts, goals, timeout)
    else:
        if storage.has_result(scenario, storage.ResultType.ICTS_INFO):
            info = storage.get_result(
                scenario, storage.ResultType.ICTS_INFO)
        else:
            info = icts(env, starts, goals, timeout)
            storage.save_result(scenario, storage.ResultType.ICTS_INFO, info)
    return info


def icts(env, starts, goals, timeout=DEFAULT_TIMEOUT_S, return_paths=False):
    info = icts_plan(env, starts, goals, timeout)
    if return_paths:
        return paths_from_info(info)
    else:
        return info

# decentralized ###############################################################


def to_agent_objects(env, starts, goals, policy=PolicyType.RANDOM):
    n_agents = starts.shape[0]
    agents = []
    for i_a in range(n_agents):
        a = Agent(env, starts[i_a])
        if not a.give_a_goal(goals[i_a]):
            return INVALID
        agents.append(a)
    return agents


def indep(env, starts, goals):
    agents = to_agent_objects(env, starts, goals)
    if agents is INVALID:
        return INVALID
    paths = []
    for a in agents:
        p = a.path
        le = p.shape[0]
        ts = np.arange(le)
        paths.append(
            np.append(p, ts.reshape((le, 1)), axis=1)
        )
    return paths


def cached_decentralized(env, starts, goals,
                         policy: PolicyType, skip_cache=False):
    if policy == PolicyType.RANDOM:
        result_type = storage.ResultType.DECEN_RANDOM
    elif policy == PolicyType.LEARNED:
        result_type = storage.ResultType.DECEN_LEARNED
    else:
        raise NotImplementedError("only random or learned policies supported")
    scenario = (env, starts, goals)
    if skip_cache:
        metrics = decentralized(env, starts, goals, policy)
    else:
        if storage.has_result(scenario, result_type):
            metrics = storage.get_result(
                scenario, result_type)
        else:
            metrics = decentralized(env, starts, goals, policy)
            storage.save_result(scenario, result_type, metrics)
    return metrics


def decentralized(env, starts, goals, policy: PolicyType):
    agents = to_agent_objects(env, starts, goals, policy)
    if agents is INVALID:
        return INVALID
    return run_a_scenario(
        env, agents, plot=False, iterator=IteratorType.BLOCKING1)
