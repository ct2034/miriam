import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from astar.astar_grid48con import astar_grid4con, distance
from planner.base import astar_base

path_save = {}


def plan(agent_pos: list, jobs: list, idle_goals: list, grid: np.array, plot: bool = False,
         filename: str = 'path_save.pkl'):
    """
    Main entry point for planner
    :param agent_pos: agent poses
    :param jobs: jobs to plan for (((s_x, s_y), (g_x, g_y)), ...)
    :param idle_goals: idle goals to consider (((g_x, g_y), (t_mu, t_std)), ...)
    :param grid: the map (2D-space + time)
    :param plot: whether to plot conditions and results or not
    :param filename: filename to save / read path_save (set to False to not do this)
    :return: tuple of tuples of agent -> job allocations, agent -> idle goal allocations and blocked map areas
    """

    # load path_save
    if filename:  # TODO: check if file was created on same map
        global path_save
        try:
            with open(filename, 'rb') as f:
                path_save = pickle.load(f)
        except FileNotFoundError:
            print("WARN: File", filename, "does not exist")

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

    _paths = get_paths(condition, comp2state(agent_job, agent_idle, blocked))

    # save path_save
    if filename:
        try:
            with open(filename, 'wb') as f:
                pickle.dump(path_save, f, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(e)

    if plot:
        # Plot input conditions
        plt.style.use('bmh')
        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax.set_aspect('equal')

        # Set grid lines to between the cells
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
        igs_array = np.array(igs)
        plt.scatter(igs_array[:, 0],
                    igs_array[:, 1],
                    s=np.full(igs_array.shape[0], 100),
                    color='g',
                    alpha=.9)

        # Legendary!
        plt.legend(["Agents", "Idle Goals"])
        plt.title("Problem Configuration and Solution")

        # plt.show()
        from mpl_toolkits.mplot3d import Axes3D
        _ = Axes3D
        ax3 = fig.add_subplot(122, projection='3d')

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


# Main methods

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

    eval_blocked = False
    for i in range(len(blocked)):
        if blocked[i][1].__class__ != int:  # two agents blocked
            eval_blocked = True
            break

    if eval_blocked:
        blocked1 = []
        blocked2 = []
        for i in range(len(blocked)):
            if blocked[i][1].__class__ != int:  # two agents blocked
                blocked1.append((blocked[i][0], blocked[i][1][0]))
                blocked2.append((blocked[i][0], blocked[i][1][1]))
            else:
                blocked1.append(blocked[i])
                blocked2.append(blocked[i])
        return [comp2state(agent_job, agent_idle, tuple(blocked1)),
                comp2state(agent_job, agent_idle, tuple(blocked2))]
    else:
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


def cost(_condition: dict, _state1: tuple, _state2: tuple) -> float:
    """
    Get the cost increase for a change from _state1 to _state2
    :param _condition: The conditions of the problem
    :param _state1: The previous state
    :param _state2: The following state
    :return: The cost increase between _state1 and _state2
    """
    (agent_pos, jobs, idle_goals, _map) = condition2comp(_condition)
    (agent_job1, agent_idle1, block_state1) = state2comp(_state1)
    (agent_job2, agent_idle2, block_state2) = state2comp(_state2)
    _cost = 0.

    block_dict1 = get_blocks_dict(block_state1)
    block_dict2 = get_blocks_dict(block_state2)
    if block_state2 != block_state1:  # blocks were added
        for aj in agent_job2:
            agent = agent_pos[aj[0]]
            block1, block2 = get_block_diff(aj[0], block_dict1, block_dict2)
            job = jobs[aj[1]]
            _cost += path_duration(path(agent, job[0], _map, block2)) ** 2 - path_duration(
                path(agent, job[0], _map, block1)) ** 2
            _cost += path_duration(path(job[0], job[1], _map, block2)) ** 2 - path_duration(
                path(agent, job[0], _map, block1)) ** 2
        for ai in agent_idle2:
            agent = agent_pos[ai[0]]
            block1, block2 = get_block_diff(agent, block_dict1, block_dict2)
            idle_goal = idle_goals[ai[1]]
            path_len = path_duration(path(agent, idle_goal[0], _map, block1)) ** 2
            # taking cumulative distribution from std, making in cheaper to arrive early
            p = norm.cdf(path_len, loc=idle_goal[1][0], scale=idle_goal[1][1])
            path_len2 = path_duration(path(agent, idle_goal[0], _map, block2)) ** 2
            p2 = norm.cdf(path_len, loc=idle_goal[1][0], scale=idle_goal[1][1])
            _cost += (p2 * path_len2) - (p * path_len)  # minus cost from previous state
    elif (agent_job2 != agent_job1 or
                  agent_idle2 != agent_idle1):
        # finding paths
        for aj in agent_job2:
            if aj not in agent_job1:
                agent = agent_pos[aj[0]]
                if agent in block_dict1.keys():
                    block = block_dict1[agent]
                else:
                    block = []
                job = jobs[aj[1]]
                _cost += (path_duration(path(agent, job[0], _map, block)) ** 2 + path_duration(
                    path(job[0], job[1], _map, block))) ** 2
        for ai in agent_idle2:
            if ai not in agent_idle1:
                agent = agent_pos[ai[0]]
                if agent in block_dict1.keys():
                    block = block_dict1[agent]
                else:
                    block = []
                idle_goal = idle_goals[ai[1]]
                path_len = path_duration(path(agent, idle_goal[0], _map, block)) ** 2
                # taking cumulative distribution from std, making in cheaper to arrive early
                p = norm.cdf(path_len, loc=idle_goal[1][0], scale=idle_goal[1][1])
                _cost += (p * path_len)
    else:
        assert False, "nothing changed in this state transition"

    # finding collisions in paths
    collision = find_collision(get_paths(_condition, _state2))
    if collision != ():
        _cost += 1
        if block_state2 == ():
            block_state2 = (collision,)
        else:
            block_state2 = block_state2 + (collision,)
        _state2 = comp2state(agent_job2, agent_idle2, block_state2)
    return _cost, _state2


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
    if n_jobs2assign == 0:
        return 0

    (agent_pos, idle_goals, jobs
     ) = clear_set(agent_idle, agent_job, agent_pos, idle_goals, jobs)

    l = []
    for j in jobs:
        p = path(j[0], j[1], _map, [], False)
        if p:  # if there was a path in the dict
            l.append(path_duration(p) ** 2)
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
                # TODO: think about this part of the heuristic. Problem is: we don't know, which agent
    return _cost


def goal_test(_condition: dict, _state: tuple) -> bool:
    """
    Test if a state is the goal state regarding given conditions
    :param _condition: Given conditions
    :param _state: State to check
    :return: Result of the test
    """
    (agent_pos, jobs, idle_goals, _) = condition2comp(_condition)
    (agent_job, agent_idle, blocked) = state2comp(_state)
    agents_blocked = False
    for i in range(len(blocked)):
        if blocked[i][1].__class__ != int:  # two agents blocked
            agents_blocked = True
            break
    return (len(agent_pos) == len(agent_job) + len(agent_idle)) and not agents_blocked


# Path Helpers

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


def path(start: tuple, goal: tuple, _map: np.array, blocked: list, calc: bool = True) -> list:
    """
    Calculate or return pre-calculated path from start to goal
    :param start: The start to start from
    :param goal: The goal to plan to
    :param _map: The map to plan on
    :param blocked: List of blocked points for agents e.g. ((x, y, t), agent)
    :param calc: whether or not the path should be calculated if no saved id available. (returns False if not saved)
    :return: the path
    """
    index = tuple([start, goal]) + tuple(blocked)
    seen = set()
    if len(blocked) > 0:
        for b in blocked:
            _map = _map.copy()
            _map[(b[1],
                  b[0],
                  b[2])] = -1
            if b in seen:
                assert False, "Duplicate blocked entries"
            seen.add(b)

    if index not in path_save.keys():
        if calc:  # if we want to calc (i.e. find the cost)
            _path = astar_grid4con((start + (0,)),
                                   (goal + (_map.shape[2] - 1,)),
                                   _map.swapaxes(0, 1))

            path_save[index] = _path
        else:
            return False
    else:
        _path = path_save[index]

    # _path = _path.copy()
    for b in blocked:
        if b in _path:
            assert False, "Path still contains the collision"
    return _path


def path_duration(_path: list) -> int:
    """Measure the duration that the traveling of a path would take
    :param _path: The path to measure
    :returns The duration"""
    return len(_path) - 1  # assuming all steps take one time unit


# Collision Helpers

def get_paths(_condition: dict, _state: tuple) -> list:
    """
    Get the path_save for a given state
    :param _condition: Input condition (
    :param _state:
    :return: tuple of path_save for agents
    """
    (agent_pos, jobs, idle_goals, _map) = condition2comp(_condition)
    (agent_job, agent_idle, blocked) = state2comp(_state)
    agent_job = np.array(agent_job)
    agent_idle = np.array(agent_idle)
    _paths = []
    blocks = get_blocks_dict(blocked)
    for ia in range(len(agent_pos)):
        if ia in blocks.keys():
            block = blocks[ia]
        else:
            block = []
        for aj in agent_job:
            if aj[0] == ia:
                p1 = path(agent_pos[ia], jobs[aj[1]][0], _map, block, calc=True)
                p1l = len(p1)
                block2 = []
                for b in block:
                    if b[2] > p1l:
                        block2.append((b[0], b[1], b[2] - p1l))
                p = concat_paths(p1.copy(),
                                 path(jobs[aj[1]][0], jobs[aj[1]][1], _map, block2, calc=True))
                _paths.append(p)
                break
        for ai in agent_idle:
            if ai[0] == ia:
                _paths.append(path(agent_pos[ia], idle_goals[ai[1]][0], _map, block, calc=True))
                break
    assert len(_paths) <= len(agent_pos), "More paths than agents"
    return _paths


def find_collision(_paths: list) -> tuple:
    """
    Find collisions in a set of path_save
    :param _paths: set of path_save
    :return: first found collisions
    """
    from_agent = []
    all_paths = []
    i = 0
    seen = set()
    for _path in _paths:
        for point in _path:
            if point in seen:  # collision
                return tuple((point, (i, from_agent[all_paths.index(point)])))
            seen.add(point)
            all_paths.append(point)
            from_agent.append(i)
        i += 1  # next path (of next agent)
    return ()


def concat_paths(path1: list, path2: list) -> list:
    """
    Append to path_save to each other
    :param path1: First path
    :param path2: Second path
    :return: Both path_save
    """
    if (path1[-1][0] == path2[0][0]) and (path1[-1][1] == path2[0][1]):
        path2.remove(path2[0])
    d = len(path1) - 1
    for i in range(len(path2)):
        path1.append((path2[i][0],
                      path2[i][1],
                      path2[i][2] + d))
    return path1


def get_blocks_dict(blocked):
    block_dict = {}
    for b in blocked:
        if b[1].__class__ == int:  # a block, not a conflict
            if b[1] in block_dict.keys():  # agent_nr
                block_dict[b[1]] += [b[0], ]
            else:
                block_dict[b[1]] = [b[0], ]
    return block_dict


def get_block_diff(agent, blocks1, blocks_new):
    if agent in blocks1.keys():
        block1 = blocks1[agent]
        block2 = blocks1[agent]
    else:
        block1 = []
        block2 = []
    if agent in blocks_new.keys():
        block2 += blocks_new[agent]
    return block1, block2


# Data Helpers

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
    return agent_job, agent_idle, blocked
