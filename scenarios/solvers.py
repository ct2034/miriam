from definitions import INVALID
from planner.policylearn.libMultiRobotPlanning.plan_ecbs import plan_in_gridmap
from planner.matteoantoniazzi_mapf.plan import icts_plan

SCHEDULE = 'schedule'


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
