import functools
import logging
import multiprocessing
import pickle
import uuid
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
from astar.astar_grid48con import astar_grid4con, distance_manhattan
from cbs_ext.base import astar_base
from scipy.stats import norm

from smartleitstand.astar.base import NoPathException

path_save = {}

def plan(agent_pos: list, jobs: list, alloc_jobs: list, idle_goals: list, grid: np.array, plot: bool = False,
         filename: str = 'path_save.pkl'):
    """
    Main entry point for planner

    Args:
      agent_pos: agent poses
      jobs: jobs to plan for (((s_x, s_y), (g_x, g_y), time), ((s_x ...), ..., time), ...)
        where time might be negative value if job is waiting
      alloc_jobs: preallocation of agent to a job (i.e. will travel to its goal)
      idle_goals: idle goals to consider (((g_x, g_y), (t_mu, t_std)), ...)
      grid: the map (2D-space + time)
      plot: whether to plot conditions and results or not
      filename: filename to save / read path_save (set to False to not do this)
      agent_pos: list: 
      jobs: list: 
      alloc_jobs: list: 
      idle_goals: list: 
      grid: np.array: 
      plot: bool:  (Default value = False)
      filename: str:  (Default value = 'path_save.pkl')

    Returns:
      : tuple of tuples of agent -> job allocations, agent -> idle goal allocations and blocked map areas

    """

    # load path_save
    if filename:  # TODO: check if file was created on same map
        load_paths(filename)

    global pool
    n = multiprocessing.cpu_count()
    if n <= 4:
        n = n * 2
    pool = multiprocessing.Pool(processes=n)

    agent_job = []

    for _ in range(len(agent_pos)):
        agent_job.append(tuple())
    for aj in alloc_jobs:
        agent_job[aj[0]] = (aj[1],)
    agent_job = tuple(agent_job)

    _agent_idle = ()
    blocked = ()
    condition = comp2condition(agent_pos, jobs, alloc_jobs, idle_goals, grid)

    # planning!
    (agent_job, _agent_idle, blocked
     ) = astar_base(start=comp2state(agent_job, _agent_idle, blocked),
                    condition=condition,
                    goal_test=goal_test,
                    get_children=get_children,
                    heuristic=heuristic,
                    cost=cost)

    _paths = get_paths(condition, comp2state(agent_job, _agent_idle, blocked))

    # save path_save
    if filename:
        save_paths(filename)

    if plot:
        fig = plot_inputs(agent_pos, idle_goals, jobs, grid)

        # plt.show()
        from mpl_toolkits.mplot3d import Axes3D
        _ = Axes3D
        ax3 = fig.add_subplot(122, projection='3d')
        ax3.axis([-1, len(grid[:, 0]), -1, len(grid[:, 0])])

        # plot agent -> job allocation
        for i_a in range(len(agent_pos)):
            for i_j in agent_job[i_a]:
                plt.arrow(x=agent_pos[i_a][0],
                          y=agent_pos[i_a][1],
                          dx=(jobs[i_j][0][0] - agent_pos[i_a][0]),
                          dy=(jobs[i_j][0][1] - agent_pos[i_a][1]),
                          ec='r',
                          fill=False,
                          linestyle='dotted')
        # plot agent -> idle goal allocations
        for ai in _agent_idle:
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

        assert _paths, "Paths have not been set"
        for _pathset in _paths:  # pathset per agent
            for p in _pathset:
                pa = np.array(p)
                ax3.plot(xs=pa[:, 0],
                         ys=pa[:, 1],
                         zs=pa[:, 2])
                legend_str.append("Agent " + str(i))
            i += 1
        plt.legend(legend_str)
        plt.title("Solution")

        plt.show()

    pool.close()

    return agent_job, _agent_idle, _paths


# Main methods

def get_children(_condition: dict, _state: tuple) -> list:
    """
    Get all following states

    Args:
      _condition: The conditions of the problem
      _state: The parent state
    to be expanded

    Returns:
      : A list of children

    """
    (agent_pos, jobs, alloc_jobs, idle_goals, _) = condition2comp(_condition)
    (agent_job, _agent_idle, blocked) = state2comp(_state)
    (left_agent_pos, left_idle_goals, left_jobs
     ) = clear_set(_agent_idle, agent_job, agent_pos, idle_goals, jobs)

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
        return [comp2state(agent_job, _agent_idle, tuple(blocked1)),
                comp2state(agent_job, _agent_idle, tuple(blocked2))]
    else:
        children = []
        agent_pos = list(agent_pos)
        agent_job = list(agent_job)
        jobs = list(jobs)
        idle_goals = list(idle_goals)
        if len(left_jobs) > 0:  # still jobs to assign - try with all left agents
            for left_job in left_jobs:  # this makes many children ...
                job_to_assign = jobs.index(left_job)
                for i_a in range(len(agent_pos)):
                    agent_job_new = agent_job.copy()
                    agent_job_new[i_a] += (job_to_assign,)
                    children.append(comp2state(tuple(agent_job_new),
                                               _agent_idle,
                                               blocked))
            return children
        elif (len(left_idle_goals) > 0) & (
                    len(left_agent_pos) > 0):  # only idle goals to assign - try with all left agents
            for i_a in range(len(left_agent_pos)):
                for i_ig in range(len(left_idle_goals)):
                    agent_idle_new = list(_agent_idle).copy()
                    agent_idle_new.append(((agent_pos.index(left_agent_pos[i_a])),
                                           idle_goals.index(left_idle_goals[i_ig])))
                    children.append(comp2state(tuple(agent_job),
                                               tuple(agent_idle_new),
                                               blocked))
            return children
        else:  # all assigned
            return []


def cost(_condition: dict, _state: tuple):
    """
    Get the cost increase for a change from _state1 to _state2

    Args:
      _condition: The conditions of the problem
      _state: The state to evaluate

    Returns:
      The **total** cost of this state
    """
    (agent_pos, jobs, alloc_jobs, idle_goals, _map) = condition2comp(_condition)
    (agent_job, _agent_idle, block_state) = state2comp(_state)
    _cost = 0.

    _paths = get_paths(_condition, _state)
    if _paths is False:
        return 99999, _state
    for i_a in range(len(_paths)):
        pathset = list(_paths[i_a])
        assigned_jobs = agent_job[i_a]
        if len(assigned_jobs) > 0:
            if (i_a, assigned_jobs[0]) in alloc_jobs:  # first entry is a preallocated job
                assert len(pathset) % 2 == 1, "Must be odd number of paths"  # since first is one only
                i = 1
                for p in pathset[2::2]:
                    _cost += p[-1][2]
                    _cost += jobs[assigned_jobs[i]][2] * -1  # waiting time before job was touched
                    i += 1
            else:
                assert len(pathset) % 2 == 0, "Must be even number of paths"
                i = 0
                for p in pathset[1::2]:
                    _cost += p[-1][2]  # each arrival time
                    _cost += jobs[assigned_jobs[i]][2] * -1  # waiting time before job was touched
                    i += 1
            assert i == len(assigned_jobs), "Not handled all assigned jobs"
        idle_assignment = list(filter(lambda x: x[0] == i_a, _agent_idle))
        if idle_assignment:
            assert len(idle_assignment) == 1, "Multiple agent entries"
            i_idle_goal = idle_assignment[0][1]
            idle_goal_stat = idle_goals[i_idle_goal][1]
            path_len = pathset[0][-1][2]  # this agent will have only one path in its set, or has it?
            assert len(pathset) == 1, "an agent with idle goal must only have one path in its set"
            prob = norm.cdf(path_len, loc=idle_goal_stat[0], scale=idle_goal_stat[1])
            _cost += (prob * path_len)

    # finding collisions in paths
    collision = find_collision(_paths)
    if collision != ():
        block_state += (collision,)
        _state = comp2state(agent_job, _agent_idle, block_state)
    for bs in block_state:
        if not bs[1].__class__ == int:  # a collision
            _cost += 1  # a little more expensive when there is a collision
        else:
            _cost += .1  # a little when there is a block
    return _cost, _state


def heuristic(_condition: dict, _state: tuple) -> float:
    """
    Estimation from this state to the goal

    Args:
      _condition: Input condition
      _state: State to eval

    Returns:
      cost heuristic for the given state
    """
    (agent_pos, jobs, alloc_jobs, idle_goals, _map) = condition2comp(_condition)
    (agent_job, _agent_idle, _) = state2comp(_state)
    _cost = 0.

    (left_agent_pos, left_idle_goals, left_jobs
     ) = clear_set(_agent_idle, agent_job, agent_pos, idle_goals, jobs)

    paths = get_paths(_condition, _state)
    if paths is False:
        return 99999  # no feasible path set

    agentposes = []
    assert len(paths) == len(agent_pos), "All agents should have paths"
    for i_agent in range(len(paths)):
        pathset = paths[i_agent]
        if pathset:
            agentposes.append(pathset[-1][-1][0:2])  # last pose of agent
        else:
            agentposes.append(agent_pos[i_agent])  # no path yet

    for i_job in range(len(left_jobs)):
        agentposes.append(jobs[i_job][1])
    valss = []
    for i_job in range(len(left_jobs)):
        valss.append({'agentposes': agentposes,
                      'job': jobs[i_job]})

    if left_agent_pos:
        for ig in left_idle_goals:
            valss.append({'agentposes': left_agent_pos,
                          'idle_goal': ig})

    # job_costs = list(pool.map(heuristic_per_job, valss))
    job_costs = list(map(heuristic_per_job, valss))
    _cost += reduce(lambda a, b: a + b, job_costs, 0)

    return _cost


def heuristic_per_job(vals):
    agentposes = vals['agentposes']
    if 'job' in vals.keys():
        job = vals['job']
        _cost = 0
        agentposes_no_self = agentposes.copy()
        agentposes_no_self.remove(job[1])
        # closest agent pose to this jobs start
        nearest_agent = get_nearest(agentposes_no_self, job[0])
        _cost += distance_no_calc(nearest_agent, job[0])
        _cost += distance_no_calc(job[0], job[1])
    elif 'idle_goal' in vals.keys():
        idle_goal = vals['idle_goal']
        _cost = 0
        nearest_agent = get_nearest(agentposes, idle_goal[0])
        path_len = distance_no_calc(nearest_agent, idle_goal[0])
        prob = norm.cdf(path_len, loc=idle_goal[1][0], scale=idle_goal[1][1])
        _cost += (prob * path_len)
    else:
        raise AssertionError("The call dictionary was built wrongly")
    return _cost


def goal_test(_condition: dict, _state: tuple) -> bool:
    """
    Test if a state is the goal state regarding given conditions

    Args:
      _condition: Given conditions
      _state: State to check

    Returns:
      Result of the test (true if goal, else false)
    """
    (agent_pos, jobs, alloc_jobs, idle_goals, _) = condition2comp(_condition)
    (agent_job, _agent_idle, blocked) = state2comp(_state)

    for i in range(len(blocked)):
        if blocked[i][1].__class__ != int:  # two agents blocked
            return False

    if len(agent_job) > 0:
        assigned_jobs = functools.reduce(lambda l, j: l + j, agent_job, tuple())
    else:
        assigned_jobs = ()
    if len(assigned_jobs) < len(jobs):  # not all jobs assigned
        return False

    agent_assigned = list(map(lambda x: len(x) > 0, agent_job))  # jobs?
    for ai in _agent_idle:
        agent_assigned[ai[0]] = True  # idle goals
    if not np.array(agent_assigned).all():  # not all agents have something to do
        return False

    # if all test are ok ..
    return True


# Path Helpers

def path(start: tuple, goal: tuple, _map: np.array, blocked: list, path_save_process: dict = {}, calc: bool = True):
    """
    Calculate or return pre-calculated path from start to goal

    Args:
      start: The start to start from
      goal: The goal to plan to
      _map: The map to plan on
      blocked: List of blocked points for agents e.g. ((x, y, t), agent)
      path_save_process: pre-processed paths are saved here
      calc: whether or not the path should be calculated if no saved id available. (returns False if not saved)

    Returns:
      the path as list of tuples
      or [] if no path found
      or False if path shouldn't have been calculated but was not saved either
    """
    seen = set()
    for b in blocked:
        if b[0:2] == start or b[0:2] == goal:
            return [], {}  # blocked at start or goal (infeasible path)

    if len(blocked) > 0:
        for b in blocked:
            _map = _map.copy()
            _map[(b[1],
                  b[0],
                  b[2])] = -1
            if b in seen:
                assert False, "Duplicate blocked entries"
            seen.add(b)

    index = tuple([start, goal]) + tuple(blocked)
    if index not in path_save.keys():
        if calc:  # if we want to calc (i.e. find the cost)
            assert len(start) == 2, "Should be called with only spatial coords"
            try:
                _path = astar_grid4con((start + (0,)),
                                       (goal + (_map.shape[2] - 1,)),
                                       _map.swapaxes(0, 1))
            except NoPathException:
                _path = []

            path_save_process[index] = _path
        else:
            return False, {}
    else:
        _path = path_save[index]

    # _path = _path.copy()
    for b in blocked:
        if b in _path:
            assert False, "Path still contains the collision"
    return _path, path_save_process


def distance_no_calc(start: tuple, goal: tuple):
    """
    Return actual path length if precalculated available, else the manhattan distance

    Args:
      start: from
      goal: to

    Returns:
      Distance
    """
    index = tuple([start, goal])
    if index in path_save.keys():
        p = path_save[index]
        return path_duration(p)
    else:
        return distance_manhattan(start, goal)


def path_duration(_path: list) -> int:
    """
    Measure the duration that the traveling of a path would take

    Args:
      _path: The path to measure

    Returns:
      The duration

    """
    return len(_path) - 1  # assuming all steps take one time unit


def concat_paths(path1: list, path2: list) -> list:
    """
    Append two paths to each other. Will keep timing of first path and assume second starts with t=0

    Args:
      path1: First path
      path2: Second path

    Returns:
      both paths after each other
    """
    assert path2[0][2] == 0, "Second path must start with t=0"

    if path1[-1][0:2] == path2[0][0:2]:
        path2.remove(path2[0])
    d = path1[-1][2]
    for i in range(len(path2)):
        path1.append((path2[i][0],
                      path2[i][1],
                      path2[i][2] + d))
    return path1


def time_shift_path(_path: list, t: int) -> list:
    """
    Shift a path to a certain time

    Args:
      _path: the path to shift
      t: time to shift by

    Returns:
      Shifted path
    """
    assert _path[0][2] == 0, "Input path should start at t=0"
    return list(map(lambda c: (c[0], c[1], c[2] + t), _path))


def pre_calc_paths(jobs, idle_goals, grid, fname=None):
    for job in jobs:
        # job distance itself
        path(job[0], job[1], grid, [], calc=True)
        # way from end to other jobs
        for next_job in jobs:
            if next_job is not job:
                path(job[1], next_job[0], grid, [], calc=True)
        # to idle goals
        for idle_goal in idle_goals:
            path(job[1], idle_goal[0], grid, [], calc=True)

    # SAVE
    if not fname:
        fname = '/tmp/paths_' + str(uuid.uuid4()) + '.pkl'
    save_paths(fname)
    return fname


# Collision Helpers


def get_paths_for_agent(vals):
    path_save_process = {}
    _agent_idle = vals['_agent_idle']
    _map = vals['_map']
    agent_job = vals['agent_job']
    agent_pos = vals['agent_pos']
    alloc_jobs = vals['alloc_jobs']
    blocks = vals['blocks']
    i_a = vals['i_a']
    idle_goals = vals['idle_goals']
    jobs = vals['jobs']
    # -------
    paths_for_agent = tuple()
    if i_a in blocks.keys():
        block = blocks[i_a]
    else:
        block = []
    assigned_jobs = agent_job[i_a]
    pose = agent_pos[i_a][0:2]
    t_shift = 0
    for job in assigned_jobs:
        if (i_a, job) in alloc_jobs:  # can be first only; need to go to goal only
            p, path_save_process = path(pose, jobs[job][1], _map, block, path_save_process, calc=True)
            if not p:
                return 0
        else:
            # trip to start
            if len(paths_for_agent) > 0:
                pose, t_shift = get_last_pose_and_t(paths_for_agent)
            block1 = time_shift_blocks(block, t_shift)
            p1, path_save_process = path(pose, jobs[job][0], _map, block1, path_save_process, calc=True)
            if not p1:
                return 0
            paths_for_agent += (time_shift_path(p1, t_shift),)
            # start to goal
            pose, t_shift = get_last_pose_and_t(paths_for_agent)
            assert pose == jobs[job][0], "Last pose should be the start"
            block2 = time_shift_blocks(block, t_shift)
            p, path_save_process = path(jobs[job][0], jobs[job][1], _map, block2, path_save_process, calc=True)
            if not p:
                return 0
        paths_for_agent += (time_shift_path(p, t_shift),)
    for ai in _agent_idle:
        if ai[0] == i_a:
            p, path_save_process = (path(agent_pos[i_a], idle_goals[ai[1]][0], _map, block, path_save_process, calc=True))
            if not p:
                return 0
            paths_for_agent += (p,)
            break  # found for this agent
    return paths_for_agent, path_save_process


def get_paths(_condition: dict, _state: tuple):
    """
    Get the path_save for a given state

    Args:
      _condition: Input condition
      _state:

    Returns:
      list of tuples per agent with all paths for this agent as lists of tuples of coords [([(..)])]
      False if one agent was not able to reach its goal
    """
    (agent_pos, jobs, alloc_jobs, idle_goals, _map) = condition2comp(_condition)
    (agent_job, _agent_idle, blocked) = state2comp(_state)
    _agent_idle = np.array(_agent_idle)
    _paths = []
    blocks = get_blocks_dict(blocked)
    valss = []
    for i_a in range(len(agent_pos)):
        valss.append({'_agent_idle': _agent_idle,
                      '_map': _map,
                      'agent_job': agent_job,
                      'agent_pos': agent_pos,
                      'alloc_jobs': alloc_jobs,
                      'blocks': blocks,
                      'i_a': i_a,
                      'idle_goals': idle_goals,
                      'jobs': jobs})
    global pool
    res = list(pool.map(get_paths_for_agent, valss))
    for r in res:
        if r is 0:
            return False
        _paths.append(r[0])
        path_save.update(r[1])
    assert len(_paths) == len(agent_pos), "More or less paths than agents"
    return _paths


def time_shift_blocks(block, t):
    blocks_for_this_agent = []
    for b in block:
        if b[2] > t:  # is after the shift (for this or later paths)
            blocks_for_this_agent.append((b[0], b[1], b[2] - t))
    return blocks_for_this_agent


def get_last_pose_and_t(paths_for_agent, ):
    last = paths_for_agent[-1][-1]
    return last[0:2], last[2] + 1


def get_nearest(points, coord):
    """
    Return closest point to coord from points
    source: https://stackoverflow.com/a/37376187/1493204
    """
    dists = [(pow(point[0] - coord[0], 2) + pow(point[1] - coord[1], 2), point)
             for point in points]  # list of (dist, point) tuples
    nearest = min(dists)
    return nearest[1]


def find_collision(_paths: list) -> tuple:
    """
    Find collisions in a set of path_save

    Args:
      _paths: set of path_save

    Returns:
      first found collision
    """
    from_agent = []
    all_paths = []
    ia = 0  # agent iterator
    seen = set()
    for _pathset in _paths:  # pathset per agent
        for _path in _pathset:
            for point in _path:
                if point in seen:  # collision
                    assert len(from_agent) == len(all_paths), "Something went wrong"
                    assert len(from_agent) == len(seen), "Something went wrong"
                    return tuple((point, (ia, from_agent[all_paths.index(point)])))
                seen.add(point)
                all_paths.append(point)
                from_agent.append(ia)
        ia += 1  # next path (of next agent)
    return ()


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

def clear_set(_agent_idle: tuple, agent_job: tuple, agent_pos: list, idle_goals: list, jobs: list) -> tuple:
    """
    Clear condition sets of agents, jobs and idle goals already assigned with each other

    Args:
      _agent_idle:
      agent_job:
      agent_pos:
      idle_goals:
      jobs:

    Returns:

    """
    cp_agent_pos = agent_pos.copy()
    cp_idle_goals = idle_goals.copy()
    cp_jobs = jobs.copy()

    for i_a in range(len(agent_pos)):
        if len(agent_job[i_a]) > 0:  # this has jobs
            cp_agent_pos.remove(agent_pos[i_a])  # remove agent
            for j in agent_job[i_a]:
                cp_jobs.remove(jobs[j])
    for ai in _agent_idle:
        cp_agent_pos.remove(agent_pos[ai[0]])
        cp_idle_goals.remove(idle_goals[ai[1]])
    return cp_agent_pos, cp_idle_goals, cp_jobs


def plot_inputs(agent_pos, idle_goals, jobs, grid, show=False, subplot=121):
    # Plot input conditions
    plt.style.use('bmh')
    fig = plt.figure()
    # ax = fig.add_subplot(121)
    ax = fig.add_subplot(subplot)
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
    plt.title("Problem Configuration")
    if show:
        plt.show()
    return fig


def condition2comp(_condition: dict):
    """
    Transform the condition dict to its components

    Args:
      _condition:

    Returns:

    """
    return (_condition["agent_pos"],
            _condition["jobs"],
            _condition["alloc_jobs"],
            _condition["idle_goals"],
            _condition["grid"])


def comp2condition(agent_pos: list,
                   jobs: list,
                   alloc_jobs: list,
                   idle_goals: list,
                   grid: np.array):
    """
    Transform condition sections into dict to use

    Args:
      agent_pos:
      jobs:
      alloc_jobs:
      idle_goals:
      grid:

    Returns:

    """
    return {
        "agent_pos": agent_pos,
        "jobs": jobs,
        "alloc_jobs": alloc_jobs,
        "idle_goals": idle_goals,
        "grid": grid
    }


def state2comp(_state: tuple) -> tuple:
    """
    Transform the state tuple to its components

    Args:
      _state:

    Returns:

    """
    return (_state[0],
            _state[1],
            _state[2])


def comp2state(agent_job: tuple,
               _agent_idle: tuple,
               blocked: tuple) -> tuple:
    """
    Transform state sections into tuple to use

    Args:
      agent_job: tuple: 
      _agent_idle: tuple:
      blocked: tuple: 

    Returns:

    """
    return agent_job, _agent_idle, blocked


def save_paths(filename):
    try:
        with open(filename, 'wb') as f:
            pickle.dump(path_save, f, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(e)


def load_paths(filename):
    global path_save
    try:
        with open(filename, 'rb') as f:
            path_save = pickle.load(f)
    except FileNotFoundError:
        logging.warning("WARN: File %s does not exist", filename)
