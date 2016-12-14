import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

import pickle

from astar.astar_grid48con import astar_grid4con, path_length, distance
from planner.base import astar_base

paths = {}

def plan(agent_pos, jobs, idle_goals, grid, plot=False, fname='paths.pkl'):
    """
    Main entry point for planner
    :param agent_pos: agent poses
    :param jobs: jobs to plan for (((s_x, s_y), (g_x, g_y)), ...)
    :param idle_goals: idle goals to consider (((g_x, g_y), (t_mu, t_std)), ...)
    :param grid: the map (2D-space + time)
    :param plot: whether to plot conditions and results or not
    :param fname: filename to save / read paths (set to False to not do this)
    :return: tuple of tuples of agent -> job allocations, agent -> idle goal allocations and blocked map areas
    """

    # load paths
    if fname:
        try:
            with open(fname, 'rb') as f:
                paths = pickle.load(f)
        except FileNotFoundError as e:
            print("WARN: File", fname, "does not exist")

    if plot:
        # Plot input conditions
        plt.style.use('bmh')
        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax.set_aspect('equal')

        # Set ticklines to between the cells
        major_ticks = np.arange(0, len(grid[:, 0, 0]) + 1, 2)
        minor_ticks = np.arange(0, len(grid[:, 0, 0]) + 1, 1) + .5
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_yticks(major_ticks)
        ax.set_yticks(minor_ticks, minor=True)
        ax.grid(which='minor', alpha=0.5)
        ax.grid(which='major', alpha=0.2)

        # Make positive y pointing up
        ax.axis([-1, len(grid[:, 0]), -1, len(grid[:, 0])])

        # Show map
        plt.imshow(grid[:, :, 0] * -1, cmap="Greys", interpolation='nearest')
        # Agents
        agents = np.array(agent_pos)
        plt.scatter(agents[:, 0],
                    agents[:, 1],
                    s=np.full(agents.shape[0], 100),
                    color='blue',
                    alpha=.9)
        # Jobs
        for j in jobs:
            plt.arrow(x=j[0][0],
                      y=j[0][1],
                      dx=(j[1][0] - j[0][0]),
                      dy=(j[1][1] - j[0][1]),
                      head_width=.3, head_length=.7,
                      length_includes_head=True,
                      ec='r',
                      fill=False)
        # Idle Goals
        igs = []
        for ai in idle_goals:
            igs.append(ai[0])
        igsa = np.array(igs)
        plt.scatter(igsa[:, 0],
                    igsa[:, 1],
                    s=np.full(igsa.shape[0], 100),
                    color='g',
                    alpha=.9)

        # Legendary!
        plt.legend(["Agents", "Idle Goals"])
        plt.title("Problem Configuration and Solution")

    # result data structures
    agent_job = ()
    agent_idle = ()
    blocked = ()
    condition = comp2condition(agent_pos, jobs, idle_goals, grid)

    # planning!
    (agent_job, agent_idle, blocked
     ) = astar_base(start=comp2state(agent_job, agent_idle, blocked),
                    condition=condition,
                    goal_test=goal_test,
                    get_children=get_children,
                    heuristic=heuristic,
                    cost=cost)

    assert len(blocked) == 0, "There are blocked points in the final state"
    _paths = get_paths(condition, comp2state(agent_job, agent_idle, blocked))

    # save paths
    if fname:
        try:
            with open(fname, 'wb') as f:
                pickle.dump(paths, f, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(e)

    if plot:
        # plot agent -> job allocation
        for aj in agent_job:
            plt.arrow(x=agent_pos[aj[0]][0],
                      y=agent_pos[aj[0]][1],
                      dx=(jobs[aj[1]][0][0] - agent_pos[aj[0]][0]),
                      dy=(jobs[aj[1]][0][1] - agent_pos[aj[0]][1]),
                      ec='r',
                      fill=False,
                      linestyle='dotted')
        # plot agent -> idle goal allocations
        for ai in agent_idle:
            plt.arrow(x=agent_pos[ai[0]][0],
                      y=agent_pos[ai[0]][1],
                      dx=(idle_goals[ai[1]][0][0] - agent_pos[ai[0]][0]),
                      dy=(idle_goals[ai[1]][0][1] - agent_pos[ai[0]][1]),
                      ec='g',
                      fill=False,
                      linestyle='dotted')

        # Paths
        from mpl_toolkits.mplot3d import Axes3D
        _ = Axes3D
        ax3 = fig.add_subplot(122, projection='3d')
        legend_str = []
        i = 0
        for p in _paths:
            pa = np.array(p)
            ax3.plot(xs=pa[:, 0],
                     ys=pa[:, 1],
                     zs=pa[:, 2])
            legend_str.append("Agent " + str(i))
            i += 1
        plt.legend(legend_str)

        plt.show()

    return agent_job, agent_idle, _paths


def get_paths(_condition: dict, _state: tuple) -> tuple:
    """
    Get the paths for a given state
    :param _condition: Input condition (
    :param _state:
    :return: tuple of paths for agents
    """
    (agent_pos, jobs, idle_goals, _map) = condition2comp(_condition)
    (agent_job, agent_idle, _) = state2comp(_state)
    agent_job = np.array(agent_job)
    agent_idle = np.array(agent_idle)
    _paths = []
    for ia in range(len(agent_pos)):
        for aj in agent_job:
            if aj[0] == ia:
                p = concat_paths(path(agent_pos[ia], jobs[aj[1]][0], _map, calc=False),
                                 path(jobs[aj[1]][0], jobs[aj[1]][1], _map, calc=False))
                _paths.append(p)
                break
        for ai in agent_idle:
            if ai[0] == ia:
                _paths.append(path(agent_pos[ia], idle_goals[ai[1]][0], _map, calc=False))
                break
    assert len(_paths) == len(agent_pos), "Not all agents have a path (or some have more)"
    return _paths


def concat_paths(path1, path2):
    """
    Append to paths to each other
    :param path1: First path
    :param path2: Second path
    :return: Both paths
    """
    if (path1[-1][0] == path2[0][0]) and (path1[-1][1] == path2[0][1]):
        path2.remove(path2[0])
    d = len(path1) - 1
    for i in range(len(path2)):
        path1.append((path2[i][0],
                      path2[i][1],
                      path2[i][2] + d))
    return path1


def heuristic(_condition: dict, _state: tuple) -> float:
    """
    Estimation from this state to the goal
    :param _condition: Input condition (
    :param _state:
    :return: cost heuristic for the given state
    """
    (agent_pos, jobs, idle_goals, _map) = condition2comp(_condition)
    (agent_job, agent_idle, _) = state2comp(_state)
    _cost = 0.

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
    if len(l) > len(agent_pos):  # we assign only jobs
        for i in range(len(agent_pos)):
            _cost += l[i]
    else:  # we have to assign idle_goals, two
        for i in range(len(agent_pos)):
            if i < len(l):
                _cost += l[i]
                # TODO: think about this part of the heuristic. Problem is: we dont know, which agent

    return _cost


def get_children(_condition: dict, _state: tuple) -> list:
    """
    Get all following states
    :param _condition: The conditions of the problem
    :param _state: The parent state
    :return: List of children
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


def clear_set(agent_idle: tuple, agent_job: tuple, agent_pos: list, idle_goals: list, jobs: list) -> tuple:
    """
    Clear condition sets of agents, jobs and idle goals already assigned with each other
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


def cost(_condition: dict, _state1: tuple, _state2: tuple) -> float:
    """
    Get the cost increase for a change from _state1 to _state2
    :param _condition: The conditions of the problem
    :param _state1: The previous state
    :param _state2: The following state
    :return: The cost increase between _state1 and _state2
    """
    (agent_pos, jobs, idle_goals, _map) = condition2comp(_condition)
    (agent_job1, agent_idle1, _) = state2comp(_state1)
    (agent_job2, agent_idle2, _) = state2comp(_state2)
    _cost = 0.
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


def path(start: tuple, goal: tuple, _map: np.array, calc: bool = True) -> list:
    """
    Calculate or return pre-calculated path from start to goal
    :param start: The start to start from
    :param goal: The goal to plan to
    :param _map: The map to plan on
    :param calc: whether or not the path should be calculated
                 if no saved id available.
                 (returns False if not saved)
    :return: the path
    """
    index = [start, goal]
    index.sort()
    reverse = index != [start, goal]

    if tuple(index) not in paths.keys():
        if calc:  # if we want to calc (i.e. find the cost)
            paths[tuple(index)] = astar_grid4con((index[0] + (0,)),
                                                 (index[1] + (_map.shape[0] * 5,)),
                                                 _map.swapaxes(0, 1))
        else:
            return False

    _path = paths[tuple(index)].copy()
    if reverse: _path.reverse()
    return _path


def goal_test(_condition: dict, _state: tuple) -> bool:
    """
    Test if a state is the goal state ragrding given conditions
    :param _condition: Given conditions
    :param _state: State to check
    :return: Result of the test
    """
    (agent_pos, jobs, idle_goals, _) = condition2comp(_condition)
    (agent_job, agent_idle, _) = state2comp(_state)
    return len(agent_pos) == len(agent_job) + len(agent_idle)


def condition2comp(_condition: dict):
    """
    Transform the condition dict to its components
    """
    return (_condition["agent_pos"],
            _condition["jobs"],
            _condition["idle_goals"],
            _condition["grid"])


def comp2condition(agent_pos: list,
                   jobs: list,
                   idle_goals: list,
                   grid: np.array):
    """
    Transform condition sections into dict to use
    """
    return {
        "agent_pos": agent_pos,
        "jobs": jobs,
        "idle_goals": idle_goals,
        "grid": grid
    }


def state2comp(_state: tuple) -> tuple:
    """
    Transform the state tuple to its components
    """
    return (_state[0],
            _state[1],
            _state[2])


def comp2state(agent_job: tuple,
               agent_idle: tuple,
               blocked: tuple) -> tuple:
    """
    Transform state sections into tuple to use
    """
    return (agent_job, agent_idle, blocked)
