import numpy as np
from definitions import INVALID
from planner.matteoantoniazzi_mapf.plan import icts_plan
from planner.policylearn.libMultiRobotPlanning.plan_ecbs import plan_in_gridmap
from sim.decentralized.agent import Agent, Policy

SCHEDULE = 'schedule'


def to_agent_objects(env, starts, goals, policy=Policy.RANDOM):
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


def ecbs(env, starts, goals, timeout=10):
    """Plan scenario using ecbs returning results data"""
    try:
        data = plan_in_gridmap(env, list(starts), list(goals), timeout)
    except KeyError:  # happens when start or goal is not in map
        return INVALID
    if data is None:
        return INVALID
    n_agents = starts.shape[0]
    schedule = data[SCHEDULE]
    assert n_agents == len(schedule.keys()), "Plans for all agents"
    return data


def icts(env, starts, goals, timeout=30):
    return icts_plan(env, starts, goals, timeout)
