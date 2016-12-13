import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from astar.astar_grid48con import astar_grid4con, path_length, distance
from planner.base import astar_base

save = {}


# condition = {
#     "agent_pos": [],
#     "jobs": [],
#     "idle_goals": [],
#     "grid": np.array([])
# }
#
# state = {
#     "agent_job": (),
#     "agent_idle": (),
#     "blocked": ()
# }


def plan(agent_pos, jobs, idle_goals, grid, plot=False):
    n_agents = len(agent_pos)

    if plot:
        plt.imshow(grid[:, :, 0] * -1, cmap="Greys", interpolation='nearest')
        agents = np.array(agent_pos)
        plt.scatter(agents[:, 0],
                    agents[:, 1],
                    s=np.full(agents.shape[0], 100),
                    color='blue',
                    alpha=.9)
        for j in jobs:
            plt.arrow(x=j[0][0],
                      y=j[0][1],
                      dx=(j[1][0] - j[0][0]),
                      dy=(j[1][1] - j[0][1]),
                      head_width=0.3, head_length=1,
                      ec='r',
                      fill=False)
        igs = []
        for ai in idle_goals:
            igs.append(ai[0])
        igsa = np.array(igs)
        plt.scatter(igsa[:, 0],
                    igsa[:, 1],
                    s=np.full(igsa.shape[0], 100),
                    color='g',
                    alpha=.9)
        plt.legend(["Agents", "Idle Goals"])
        plt.title("Problem Configuration and Solution")
    agent_job = ()
    agent_idle = ()
    blocked = ()

    (agent_job, agent_idle, blocked
     ) = astar_base(start=comp2state(agent_job, agent_idle, blocked),
                    condition=comp2condition(agent_pos, jobs, idle_goals, grid),
                    goal_test=goal_test,
                    get_children=get_children,
                    heuristic=heuristic,
                    cost=cost)

    if plot:
        for aj in agent_job:
            plt.arrow(x=agent_pos[aj[0]][0],
                      y=agent_pos[aj[0]][1],
                      dx=(jobs[aj[1]][0][0] - agent_pos[aj[0]][0]),
                      dy=(jobs[aj[1]][0][1] - agent_pos[aj[0]][1]),
                      ec='r',
                      fill=False,
                      linestyle='dotted')
        for ai in agent_idle:
            plt.arrow(x=agent_pos[ai[0]][0],
                      y=agent_pos[ai[0]][1],
                      dx=(idle_goals[ai[1]][0][0] - agent_pos[ai[0]][0]),
                      dy=(idle_goals[ai[1]][0][1] - agent_pos[ai[0]][1]),
                      ec='g',
                      fill=False,
                      linestyle='dotted')
        plt.show()

    # TODO: also give out paths!
    return agent_job, agent_idle, blocked

def heuristic(_condition, _state):
    """
    Estimation from this state to the goal
    """
    (agent_pos, jobs, idle_goals, _map) = condition2comp(_condition)
    (agent_job, agent_idle, _) = state2comp(_state)
    _cost = 0

    # what to assign
    n_jobs2assign = len(agent_pos) - len(agent_job)
    if n_jobs2assign == 0: return 0

    (agent_pos, idle_goals, jobs
     ) = clear_set(agent_idle, agent_job, agent_pos, idle_goals, jobs)

    l = []
    for j in jobs:
        p = path(j[0], j[1], _map, False)
        if p:  # if there was a path in the dict
            l.append(path_length(p))
        else:
            l.append(distance(j[0], j[1]))

    l.sort()
    if __name__ == '__main__':
        if len(l) > len(agent_pos):  # we assign only jobs
            for i in range(len(agent_pos)):
                _cost += l[i]
        else:  # we have to assign idle_goals, two
            pass
            # TODO: think about this part of the heuristic. Problem is: we dont know, which agent

    return _cost

def get_children(_condition, _state):
    """
    Get all following states
    """
    (agent_pos, jobs, idle_goals, _) = condition2comp(_condition)
    (agent_job, agent_idle, blocked) = state2comp(_state)
    (left_agent_pos, left_idle_goals, left_jobs
     ) = clear_set(agent_idle, agent_job, agent_pos, idle_goals, jobs)
    children = []
    agent_pos = list(agent_pos)
    jobs = list(jobs)
    idle_goals = list(idle_goals)
    if len(left_jobs) > 0:  # still jobs to assign - try with all left agents
        # TODO: what if there are too many jobs?
        for a in left_agent_pos:
            l = list(agent_job).copy()
            l.append((agent_pos.index(a),
                      jobs.index(left_jobs[0])))
            children.append(comp2state(tuple(l),
                                       agent_idle,
                                       blocked))
        return children
    elif len(left_idle_goals) > 0:  # only idle goals to assign - try with all left agents
        for a in left_agent_pos:
            l = list(agent_idle).copy()
            l.append((agent_pos.index(a),
                      idle_goals.index(left_idle_goals[0])))
            children.append(comp2state(agent_job,
                                       tuple(l),
                                       blocked))
        return children
    else:  # all assigned
        return []

def clear_set(agent_idle, agent_job, agent_pos, idle_goals, jobs):
    """
    Clear condition sets of agents, jobs and idle goals already taken care or
    """
    cp_agent_pos = agent_pos.copy()
    cp_idle_goals = idle_goals.copy()
    cp_jobs = jobs.copy()

    for aj in agent_job:
        cp_agent_pos.remove(agent_pos[aj[0]])
        cp_jobs.remove(jobs[aj[1]])
    for ai in agent_idle:
        cp_agent_pos.remove(agent_pos[ai[0]])
        cp_idle_goals.remove(idle_goals[ai[1]])
    # TODO: sort by lengths, i.e. metric of assignment order
    return cp_agent_pos, cp_idle_goals, cp_jobs

def cost(_condition, _state1, _state2):
    (agent_pos, jobs, idle_goals, _map) = condition2comp(_condition)
    (agent_job1, agent_idle1, _) = state2comp(_state1)
    (agent_job2, agent_idle2, _) = state2comp(_state2)
    _cost = 0
    for aj in agent_job2:
        if aj not in agent_job1:
            agent = agent_pos[aj[0]]
            job = jobs[aj[1]]
            _cost += path_length(path(agent, job[0], _map))
            _cost += path_length(path(job[0], job[1], _map))
    for ai in agent_idle2:
        if ai not in agent_idle1:
            agent = agent_pos[ai[0]]
            goal = idle_goals[ai[1]]
            path_len = path_length(path(agent, goal[0], _map))
            # taking cumulative distribution from std, making in cheaper to arrive early
            p = norm.cdf(path_len, loc=goal[1][0], scale=goal[1][1])
            _cost += (p * path_len)
    return _cost

def path(start: tuple, goal: tuple, _map: np.array, calc=True) -> list:
    index = [start, goal]
    index.sort()
    reverse = index != [start, goal]

    if tuple(index) not in save.keys():
        if calc:  # if we want to calc (i.e. find the cost)
            save[tuple(index)] = astar_grid4con((index[0] + (0,)), (index[1] + (_map.shape[0] * 5,)), _map)
        else:
            return False

    _path = save[tuple(index)].copy()
    if reverse: _path.reverse()
    return _path

def condition2comp(_condition: dict):
    return (_condition["agent_pos"],
            _condition["jobs"],
            _condition["idle_goals"],
            _condition["grid"])

def comp2condition(agent_pos: list,
                   jobs: list,
                   idle_goals: list,
                   grid: np.array):
    return {
        "agent_pos": agent_pos,
        "jobs": jobs,
        "idle_goals": idle_goals,
        "grid": grid
    }

def state2comp(_state: tuple) -> tuple:
    return (_state[0],
            _state[1],
            _state[2])

def comp2state(agent_job: tuple,
               agent_idle: tuple,
               blocked: tuple) -> tuple:
    return (agent_job, agent_idle, blocked)

def goal_test(_condition, current):
    (agent_pos, jobs, idle_goals, _) = condition2comp(_condition)
    (agent_job, agent_idle, _) = state2comp(current)
    return len(agent_pos) == len(agent_job) + len(agent_idle)
