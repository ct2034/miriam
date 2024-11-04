import logging
import multiprocessing
import pickle
import uuid
from functools import reduce
from itertools import product
from typing import Any, Iterator, List, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from planner.astar.astar_grid48con import distance_manhattan
from planner.common import *
from planner.eval.display import plot_inputs, plot_results
from planner.tcbs.base import MAX_COST, astar_base
from planner.tcbs.stats import StatsContainer
from tools import ColoredLogger

logging.setLoggerClass(ColoredLogger)

plt.style.use("bmh")

_config = {}
_distances = None
_stats = StatsContainer()

EXPECTED_MAX_N_BLOCKS = 1000


def plan(
    agent_pos: list,
    jobs: list,
    allocated_jobs: list,
    idle_goals: list,
    grid: np.array,
    config: dict = {},
    plot: bool = False,
    fixed_assignment=False,
):
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
      pathplanning_only_assignment: bool: do the pathplanning only (this assumes each job to the same index agent)

    Returns:
      : tuple of tuples of agent -> job allocations, agent -> idle goal allocations and blocked map areas

    """

    if not config:
        config = generate_config()  # default config
    filename = config["filename_pathsave"]
    global _config
    _config = config

    # stats
    if _config["save_stats"]:
        global _stats
        _stats = StatsContainer()

    # load path_save
    if filename:  # TODO: check if file was created on same map
        load_paths(filename)

    pool = alloc_threads()
    # alloc_threads()
    agent_job = []
    _agent_idle = []

    if _config["number_nearest"] != 0:
        global _distances
        filename = pre_calc_paths(jobs, idle_goals, grid, filename)
        _distances = pre_calc_distances(agent_pos, jobs, idle_goals, grid, filename)

    agent_pos_test = set()
    for a in agent_pos:
        # init agent_job allocation
        agent_job.append(tuple())
        _agent_idle.append(tuple())
        # check for duplicate agents
        assert a not in agent_pos_test, "Duplicate agent poses"
        agent_pos_test.add(a)

    if fixed_assignment:
        logging.info("Pathplanning only!")
        agent_job = fixed_assignment
        # for agent in range(len(agent_job)):
        #     if len(agent_job[agent]) == 0:  # no assignment
        #         new_job = ((0, 0), agent_pos[agent], 0)  # fake job
        #         jobs.append(
        #             new_job
        #         )
        #         alloc_jobs.append(
        #             (agent, jobs.index(new_job))
        #         )

    for aj in allocated_jobs:
        agent_job[aj[0]] = (aj[1],)

    agent_job = tuple(agent_job)
    _agent_idle = tuple(_agent_idle)

    # making jobs unique
    jobs = make_unique(jobs)

    blocked = ()
    condition = comp2condition(agent_pos, jobs, allocated_jobs, idle_goals, grid)

    # planning!
    (agent_job, _agent_idle, blocked) = astar_base(
        start=comp2state(agent_job, _agent_idle, blocked),
        condition=condition,
        goal_test=goal_test,
        get_children=get_children,
        heuristic=heuristic,
        cost=cost,
    )

    _paths = get_paths(condition, comp2state(agent_job, _agent_idle, blocked))

    # save path_save
    if filename:
        save_paths(filename)

    if plot:
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        plot_inputs(ax1, agent_pos, idle_goals, jobs, grid)
        ax2 = fig.add_subplot(122, projection="3d")
        plot_results(
            ax2, _agent_idle, _paths, agent_job, agent_pos, grid, idle_goals, jobs
        )
        plt.show()

    pool.close()
    pool.terminate()
    pool.join()

    if _config["save_stats"]:
        return agent_job, _agent_idle, _paths, _stats.as_dict()
    else:
        return agent_job, _agent_idle, _paths


def alloc_threads():
    global pool
    n = multiprocessing.cpu_count()
    n = min(n, 8)
    pool = multiprocessing.Pool(processes=n)
    return pool


# Main methods


def get_children(_condition: dict, _state: tuple) -> list:
    """
    Get all following states.

    Args:
      _condition: The conditions of the problem
      _state: The parent state
    to be expanded

    Returns:
      : A list of children
    """
    (agent_pos, jobs, alloc_jobs, idle_goals, _) = condition2comp(_condition)
    (agent_job, agent_idle, blocked) = state2comp(_state)
    (left_agent_pos, left_idle_goals, left_jobs) = clear_set(
        agent_idle, agent_job, agent_pos, idle_goals, jobs
    )

    eval_confilct_to_block = False
    if blocked:
        if max(map(is_conflict_not_block, blocked)) & len(left_jobs) == 0:
            eval_confilct_to_block = True

    if eval_confilct_to_block:  # a block to expand
        blocked1 = []
        blocked2 = []
        for i in range(len(blocked)):
            if is_conflict_not_block(blocked[i]):  # two agents blocked
                blocked1.append((blocked[i][:-1], blocked[i][-1][0]))
                blocked2.append((blocked[i][:-1], blocked[i][-1][1]))
            else:
                blocked1.append(blocked[i])
                blocked2.append(blocked[i])
        return [
            comp2state(agent_job, agent_idle, tuple(blocked1)),
            comp2state(agent_job, agent_idle, tuple(blocked2)),
        ]
    else:
        agent_pos = list(agent_pos)
        agent_job = list(agent_job)
        jobs = list(jobs)
        idle_goals = list(idle_goals)
        if len(left_jobs) > 0:  # still jobs to assign - try with all left agents
            if _config["number_nearest"] != 0:
                n = _config["number_nearest"]
                return assign_nearest_jobs(
                    agent_idle, agent_job, agent_pos, blocked, jobs, left_jobs, n
                )
            else:
                return assign_jobs(
                    agent_idle, agent_job, agent_pos, blocked, jobs, left_jobs
                )
        # only idle goals if more agents than jobs
        elif (len(left_idle_goals) > 0) & (len(left_jobs) == 0):
            return assign_idle_goals(
                agent_idle,
                agent_job,
                agent_pos,
                blocked,
                idle_goals,
                left_agent_pos,
                left_idle_goals,
            )
        else:  # all assigned
            return []


def assign_idle_goals(
    agent_idle,
    agent_job,
    agent_pos,
    blocked,
    idle_goals,
    left_agent_pos,
    left_idle_goals,
):
    children = []
    agent_idle = list(agent_idle)
    for i_la in range(len(left_agent_pos)):
        # which agent is it actually?
        i_a = agent_pos.index(left_agent_pos[i_la])
        if not len(agent_idle[i_a]):  # no idle goal yet
            for i_ig in range(len(left_idle_goals)):
                agent_idle_new = agent_idle.copy()
                agent_idle_new[i_a] = (idle_goals.index(left_idle_goals[i_ig]),)
                children.append(
                    comp2state(tuple(agent_job), tuple(agent_idle_new), blocked)
                )
    return children


def assign_jobs(agent_idle, agent_job, agent_pos, blocked, jobs, left_jobs):
    children = []
    for left_job in left_jobs:  # this makes many children ...
        job_to_assign = jobs.index(left_job)
        for i_a in range(len(agent_pos)):
            if not _config["time_extended"]:
                if len(agent_job[i_a]) > 0:  # has assignment
                    continue
            agent_job_new = agent_job.copy()
            agent_job_new[i_a] += (job_to_assign,)
            children.append(comp2state(tuple(agent_job_new), agent_idle, blocked))
    return children


def assign_nearest_jobs(agent_idle, agent_job, agent_pos, blocked, jobs, left_jobs, n):
    from pyflann import FLANN

    children = []
    starts = []
    ends = []
    ends_job = []
    for left_job in left_jobs:  # this makes many children ...
        ends.append(left_job[0])
        ends_job.append(jobs.index(left_job))
    for i_a in range(len(agent_pos)):
        if agent_job[i_a]:  # has assignment
            i_j = agent_job[i_a][-1]
            starts.append(jobs[i_j][0])
        else:
            starts.append(agent_pos[i_a])
    flann = FLANN()
    result, dists = flann.nn(
        np.array(ends, dtype=float),
        np.array(starts, dtype=float),
        (n if len(ends) >= n else len(ends)),
        algorithm="kmeans",
        branching=32,
        iterations=7,
        checks=16,
    )
    assert len(agent_pos) == len(result), "Not the right amount of results"
    for i_a in range(len(agent_pos)):
        if len(result.shape) == 1:
            result = np.array(
                list(
                    map(
                        lambda x: [
                            x,
                        ],
                        result,
                    )
                )
            )
        for res in result[i_a]:
            agent_job_new = agent_job.copy()
            agent_job_new[i_a] += (ends_job[res],)
            children.append(comp2state(tuple(agent_job_new), agent_idle, blocked))
    return children


def cost(_condition: dict, _state: tuple):
    """
    Get the cost for this state

    Args:
      _condition: The conditions of the problem
      _state: The state to evaluate

    Returns:
      The **total** cost of this state
    """
    (agent_pos, jobs, alloc_jobs, idle_goals, _map) = condition2comp(_condition)
    (agent_job, agent_idle, block_state) = state2comp(_state)
    _cost = 0.0

    global _config
    global _stats
    if _config["save_stats"]:
        _stats.inc_high_level_expanded()

    _paths = get_paths(_condition, _state)
    if _paths == False:  # one path was not viable
        return MAX_COST, _state
    for i_a in range(len(_paths)):
        pathset = list(_paths[i_a])
        assigned_jobs = agent_job[i_a]
        if len(assigned_jobs) > 0:
            # first entry is a preallocated job
            if (i_a, assigned_jobs[0]) in alloc_jobs:
                i = 1
                for p in pathset[2::2]:
                    _cost += p[-1][2]
                    i += 1
            else:
                i = 0
                for p in pathset[1::2]:
                    _cost += p[-1][2]  # each arrival time
                    i += 1
            assert i == len(assigned_jobs), "Not handled all assigned jobs"
        idle_assignment = agent_idle[i_a]
        if idle_assignment:
            assert len(idle_assignment) == 1, "Multiple agent entries"
            i_idle_goal = idle_assignment[0]
            idle_goal_stat = idle_goals[i_idle_goal][1]
            # this agent will have only one path in its set, or has it?
            path_len = pathset[0][-1][2]
            prob = norm.cdf(path_len, loc=idle_goal_stat[0], scale=idle_goal_stat[1])
            _cost += prob * path_len

    # finding collisions in paths
    collisions = find_collision(_paths, _config["all_collisions"])
    for collision in collisions:
        if collision != ():
            block_state += (collision,)

    seen = set()
    for b in block_state:
        if b in seen:
            return MAX_COST, _state
        seen.add(b)

    # _cost += block_state.__len__() / EXPECTED_MAX_N_BLOCKS
    # assert block_state.__len__() < EXPECTED_MAX_N_BLOCKS, "more blocks than we expected"

    _state = comp2state(agent_job, agent_idle, block_state)
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
    (agent_job, _agent_idle, block_state) = state2comp(_state)
    _cost = 0.0

    (left_agent_pos, left_idle_goals, left_jobs) = clear_set(
        _agent_idle, agent_job, agent_pos, idle_goals, jobs
    )

    paths = get_paths(_condition, _state)
    if paths is False:
        return MAX_COST  # no feasible path set

    agentposes = []
    assert len(paths) == len(agent_pos), "All agents should have paths"
    for i_agent in range(len(paths)):
        pathset = paths[i_agent]
        if pathset and _config["time_extended"]:
            agentposes.append(pathset[-1][-1][0:2])  # last pose of agent
        else:
            agentposes.append(agent_pos[i_agent])  # no path yet

    for i_lj in range(len(left_jobs)):
        agentposes.append(left_jobs[i_lj][1])
    valss = []
    for i_lj in range(len(left_jobs)):
        valss.append({"agentposes": agentposes, "job": left_jobs[i_lj]})

    if left_agent_pos:
        for ig in left_idle_goals:
            valss.append({"agentposes": left_agent_pos, "idle_goal": ig})

    job_costs = list(map(heuristic_per_job, valss))
    _cost += reduce(lambda a, b: a + b, job_costs, 0)

    _cost += block_state.__len__() / EXPECTED_MAX_N_BLOCKS
    assert block_state.__len__() < EXPECTED_MAX_N_BLOCKS, "more blocks than we expected"

    return _cost


def heuristic_per_job(vals):
    agentposes = vals["agentposes"]
    if "job" in vals.keys():
        job = vals["job"]
        _cost = 0
        agentposes_no_self = agentposes.copy()
        agentposes_no_self.remove(job[1])
        # closest agent pose to this jobs start
        nearest_agent = get_nearest(agentposes_no_self, job[0])
        _cost += distance_no_calc(nearest_agent, job[0])
        _cost += distance_no_calc(job[0], job[1])
    elif "idle_goal" in vals.keys():
        idle_goal = vals["idle_goal"]
        _cost = 0
        nearest_agent = get_nearest(agentposes, idle_goal[0])
        path_len = distance_no_calc(nearest_agent, idle_goal[0])
        prob = norm.cdf(path_len, loc=idle_goal[1][0], scale=idle_goal[1][1])
        _cost += prob * path_len
    else:
        raise AssertionError("The call dictionary was built wrong")
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
    (agent_job, agent_idle, blocked) = state2comp(_state)

    for b in blocked:
        if is_conflict_not_block(b):  # two agents blocked
            return False

    (left_agent_pos, left_idle_goals, left_jobs) = clear_set(
        agent_idle, agent_job, agent_pos, idle_goals, jobs
    )

    if len(left_jobs) > 0:
        return False

    if len(left_idle_goals) > 0:
        agent_assigned = list(map(lambda x: len(x) > 0, agent_job))  # jobs?
        for i_a in range(len(agent_idle)):
            if len(agent_idle[i_a]):
                agent_assigned[i_a] = True  # idle goals
        # not all agents have something to do and idle goals left
        if not np.array(agent_assigned).all():
            return False

    # if all test are ok ..
    return True


# Path Helpers


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
        path1.append((path2[i][0], path2[i][1], path2[i][2] + d))
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


def reverse_path(path: list) -> list:
    out = []
    i = 0
    for n in path[::-1]:
        out.append((n[0], n[1], i))
        i += 1
    return out


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
        fname = "/tmp/paths_" + str(uuid.uuid4()) + ".pkl"
    save_paths(fname)
    return fname


def pre_calc_distances(agents, tasks, idle_goals, grid, fname=None):
    dist_at = np.zeros([len(agents), len(tasks)])
    for ia, it in product(range(len(agents)), range(len(tasks))):
        dist_at[ia, it] = distance_no_calc(agents[ia], tasks[it][0])
    # tasks
    dist_t = np.zeros([len(tasks)])
    for it in range(len(tasks)):
        dist_t[it] = distance_no_calc(tasks[it][0], tasks[it][1])
    # tasks-tasks
    dist_tt = np.zeros([len(tasks), len(tasks)])
    for it1, it2 in product(range(len(tasks)), range(len(tasks))):
        dist_tt[it1, it2] = distance_no_calc(tasks[it1][1], tasks[it2][0])

    return {"at": dist_at, "t": dist_t, "tt": dist_tt}


# Collision Helpers


def get_paths_for_agent(vals):
    path_save_process = {}
    _agent_idle = vals["_agent_idle"]
    _map = vals["_map"]
    agent_job = vals["agent_job"]
    agent_pos = vals["agent_pos"]
    alloc_jobs = vals["alloc_jobs"]
    blocks = vals["blocks"]
    i_a = vals["i_a"]
    idle_goals = vals["idle_goals"]
    jobs = vals["jobs"]
    # -------
    paths_for_agent = tuple()
    if i_a in blocks.keys():
        block = blocks[i_a]
    else:
        block = []
    assigned_jobs = agent_job[i_a]
    pose = agent_pos[i_a][0:2]
    t_shift = 0
    n_expanded = 0
    for ij in assigned_jobs:
        if (i_a, ij) in alloc_jobs:  # can be first only; need to go to goal only
            p, path_save_process, exp = path(
                pose, jobs[ij][1], _map, block, path_save_process, calc=True
            )
            n_expanded += exp
            if not p:
                return False
        else:
            # trip to start
            if len(paths_for_agent) > 0:  # no route yet -> keep pose
                pose, t_shift = get_last_pose_and_t(paths_for_agent)
            block1 = time_shift_blocks(block, t_shift)
            p1, path_save_process, exp = path(
                pose, jobs[ij][0], _map, block1, path_save_process, calc=True
            )
            n_expanded += exp
            if not p1:
                return False
            paths_for_agent += (time_shift_path(p1, t_shift),)
            # start to goal
            pose, t_shift = get_last_pose_and_t(paths_for_agent)
            assert pose == jobs[ij][0], "Last pose should be the start"
            block2 = time_shift_blocks(block, t_shift)
            p, path_save_process, exp = path(
                jobs[ij][0], jobs[ij][1], _map, block2, path_save_process, calc=True
            )
            n_expanded += exp
            if not p:
                return False
        paths_for_agent += (time_shift_path(p, t_shift),)
    if len(_agent_idle[i_a]):
        p, path_save_process, exp = path(
            agent_pos[i_a],
            idle_goals[_agent_idle[i_a][0]][0],
            _map,
            block,
            path_save_process,
            calc=True,
        )
        n_expanded += exp
        if not p:
            return False
        paths_for_agent += (p,)
    return paths_for_agent, path_save_process, n_expanded


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
    _agent_idle = np.array(_agent_idle, dtype=object)
    _paths = []
    blocks = get_blocks_dict(blocked)
    valss = []
    for i_a in range(len(agent_pos)):
        valss.append(
            {
                "_agent_idle": _agent_idle,
                "_map": _map,
                "agent_job": agent_job,
                "agent_pos": agent_pos,
                "alloc_jobs": alloc_jobs,
                "blocks": blocks,
                "i_a": i_a,
                "idle_goals": idle_goals,
                "jobs": jobs,
            }
        )
    global pool
    # res_out = pool.map(get_paths_for_agent, valss)
    res_out = map(get_paths_for_agent, valss)
    res = list(res_out)
    for r in res:
        if not r:
            return False

        # TODO: check at runtime if debugging
        # debug_time_jump_in_paths([r[0]])

        _paths.append(r[0])
        path_save.update(r[1])
        _stats.add_low_level_expanded(r[2])
    longest = max(map(lambda p: len(reduce(lambda a, b: a + b, p, [])), _paths))
    if _config["finished_agents_block"]:
        (left_agent_pos, left_idle_goals, left_jobs) = clear_set(
            _agent_idle, agent_job, agent_pos, idle_goals, jobs
        )
        if left_jobs.__len__() == 0:  # Only fill up if all jobs are assigned.
            _paths = fill_up_paths(longest, _paths, agent_pos, blocks)

    # debug_time_jump_in_paths(_paths)

    assert len(_paths) == len(agent_pos), "More or less paths than agents"
    return _paths


def fill_up_paths(longest: int, _paths: list, agent_pos: list, blocks: list) -> list:
    if longest > 0:
        res_paths = []
        for ia, paths_for_agent in enumerate(_paths):
            blocks_for_agent = (
                map(lambda x: x[1], filter(lambda x: x[0] == VERTEX, blocks[ia]))
                if ia in blocks
                else []
            )
            if paths_for_agent:
                last = paths_for_agent[-1][-1]
            else:
                last = agent_pos[ia] + (-1,)
            length = len(reduce(lambda a, b: a + b, paths_for_agent, []))
            ts = range(last[2] + 1, longest)
            if ts:
                standing_section = list(map(lambda x: last[0:2] + (x,), ts))
                if standing_section:
                    paths_for_agent += (standing_section,)
            res_paths.append(paths_for_agent)
        assert len(res_paths) == len(_paths), "Not all paths processed"

        # debug_time_jump_in_paths(res_paths)
        return res_paths
    else:
        return _paths


def time_shift_blocks(blocks, t):
    blocks_for_this_agent = []
    for b in blocks:
        v = b[1]
        if v[2] >= t:  # is after the shift (for this or later paths)
            blocks_for_this_agent.append((b[0], (v[0], v[1], v[2] - t)))

    return blocks_for_this_agent


def get_last_pose_and_t(paths_for_agent):
    last = paths_for_agent[-1][-1]
    return last[0:2], last[2] + 1


def get_nearest(points, coord):
    """
    Return closest point to coord from points
    source: https://stackoverflow.com/a/37376187/1493204
    """
    dists = [
        (pow(point[0] - coord[0], 2) + pow(point[1] - coord[1], 2), point)
        for point in points
    ]  # list of (dist, point) tuples
    nearest = min(dists)
    return nearest[1]


def find_collision(_paths: list, all_collisions=False) -> tuple:
    """
    Find collisions in a set of paths. Will return vortex or edge

    Args:
      _paths: set of path_save

    Returns:
      first found vortex or edge
    """
    vortexes = {}
    edges = {}
    agent = 0
    collisions = []
    for agent_paths in _paths:
        if all(map(lambda x: len(x) == 0, agent_paths)):
            return ()  # no paths -> no collision
        elif len(agent_paths) > 1:
            path = reduce(lambda a, b: a + b, agent_paths)
        else:
            path = agent_paths[0]
        for i in range(len(path)):
            vortex = path[i][:2]
            edge = None
            if i + 1 < len(path):
                a, b = path[i][:2], path[i + 1][:2]
                edge = (a, b) if a > b else (b, a)
            t = path[i][2]
            # DEBUG: check if jump in timeline
            # if i > 0:
            #    assert path[i][2] - path[i-1][2] == 1
            if t in vortexes.keys():
                if vortex in vortexes[t].keys():  # it is already someone there
                    col = VERTEX, vortex + (t,), (agent, vortexes[t][vortex])
                    if not all_collisions:
                        return [
                            col,
                        ]
                    else:
                        collisions.append(col)
                else:
                    vortexes[t][vortex] = agent
            else:
                vortexes[t] = {}
                vortexes[t][vortex] = agent
            if edge:
                if t in edges.keys():
                    if edge in edges[t].keys():
                        col = EDGE, edge + (t,), (agent, edges[t][edge])
                        if not all_collisions:
                            return [
                                col,
                            ]
                        else:
                            collisions.append(col)
                    else:
                        edges[t][edge] = agent
                else:
                    edges[t] = {}
                    edges[t][edge] = agent
        agent += 1
    if not all_collisions:
        return ()
    else:
        return collisions


def is_conflict_not_block(blocked_i):
    return blocked_i[-1].__class__ != int


def get_blocks_dict(blocked):
    block_dict = {}
    for b in blocked:
        if not is_conflict_not_block(b):  # a block, not a conflict
            agent = b[-1]
            if agent in block_dict.keys():  # agent_nr
                block_dict[agent] += list(
                    b[:1],
                )
            else:
                block_dict[agent] = list(
                    b[:1],
                )
    return block_dict


# Data Helpers


def clear_set(
    _agent_idle: tuple, agent_job: tuple, agent_pos: list, idle_goals: list, jobs: list
) -> tuple:
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
        if len(agent_job[i_a]):  # this has jobs
            cp_agent_pos.remove(agent_pos[i_a])  # remove agent
            for j in agent_job[i_a]:
                cp_jobs.remove(jobs[j])
        if len(_agent_idle[i_a]):  # an idle goal assigned
            cp_agent_pos.remove(agent_pos[i_a])
            cp_idle_goals.remove(idle_goals[_agent_idle[i_a][0]])
    return cp_agent_pos, cp_idle_goals, cp_jobs


def make_unique(jobs):
    jobs_copy = jobs.copy()
    jobs = []
    jobs_set = set()
    for j in jobs_copy:
        while j in jobs_set:
            j = (j[0], j[1], j[2] + 0.0001)
        jobs.append(j)
        jobs_set.add(j)
    return jobs


def condition2comp(_condition: dict):
    """
    Transform the condition dict to its components

    Args:
      _condition:

    Returns:

    """
    return (
        _condition["agent_pos"],
        _condition["jobs"],
        _condition["alloc_jobs"],
        _condition["idle_goals"],
        _condition["grid"],
    )


def comp2condition(
    agent_pos: list, jobs: list, alloc_jobs: list, idle_goals: list, grid: np.array
):
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
        "grid": grid,
    }


def state2comp(_state: tuple) -> tuple:
    """
    Transform the state tuple to its components

    Args:
      _state:

    Returns:

    """
    return (_state[0], _state[1], _state[2])


def comp2state(agent_job: tuple, _agent_idle: tuple, blocked: tuple) -> tuple:
    """
    Transform state sections into tuple to use

    Args:
      agent_job: tuple:
      _agent_idle: tuple:
      blocked: tuple:

    Returns:
      the state tuple
    """
    return agent_job, _agent_idle, blocked


def generate_config():
    return {
        "filename_pathsave": "path_save.pkl",  # filename to save path cache
        # weather finished agents stand around and block others
        "finished_agents_block": False,
        "number_nearest": 0,  # 0 means all, otherwise only n nearest possible agents are checked
        # whether to insert all collisions as block (suboptimal)
        "all_collisions": False,
        # whether to use heuristic collisions resolution (suboptimal)
        "heuristic_colission": False,
        # whether to use time-extended assignments (agents can have multiple jobs)
        "time_extended": True,
        # whether to save stats
        "save_stats": False,
    }


def save_paths(filename):
    try:
        with open(filename, "wb") as f:
            pickle.dump(path_save, f, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        logging.error("Error saving paths: " + str(e))


def load_paths(filename):
    global path_save
    try:
        with open(filename, "rb") as f:
            path_save = pickle.load(f)
    except FileNotFoundError:
        logging.warning("File %s does not exist", filename)


def debug_time_jump_in_paths(paths):
    for agent_paths in paths:
        t = -1
        for a_path in agent_paths:
            for pose in a_path:
                assert pose[2] - t == 1
                t = pose[2]
