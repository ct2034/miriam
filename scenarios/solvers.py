import numpy as np
from definitions import DEFAULT_TIMEOUT_S, INVALID
from planner.matteoantoniazzi_mapf.plan import icts_plan, paths_from_info
from planner.policylearn.libMultiRobotPlanning.plan_ecbs import plan_in_gridmap
from sim.decentralized.agent import Agent
from sim.decentralized.policy import PolicyType

SCHEDULE = 'schedule'
AGENT = 'agent'


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
    schedule = data[SCHEDULE]
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


def icts(env, starts, goals, timeout=DEFAULT_TIMEOUT_S, return_paths=False):
    info = icts_plan(env, starts, goals, timeout)
    if return_paths:
        return paths_from_info(info)
    else:
        return info
