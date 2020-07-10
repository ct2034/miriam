from __future__ import division

from planner.tcbs.plan import plan as plan_cbsext, generate_config
from pyomo.opt import SolverFactory
from pyomo.core import *
from pyomo.environ import *
from itertools import *
import numpy as np
import logging


logging.getLogger('pyomo.core').setLevel(logging.ERROR)
logging.getLogger('pyomo.util').setLevel(logging.ERROR)
logging.getLogger('pyomo.util.timing.construction').setLevel(logging.ERROR)
logging.getLogger('git.cmd').setLevel(logging.ERROR)
logging.getLogger(
    'pyutilib.component.core.pyutilib.autotest').setLevel(logging.ERROR)
logging.getLogger(
    'pyutilib.component.core.pyutilib.workflow').setLevel(logging.ERROR)
logging.getLogger('pyutilib.component.core.pyutilib').setLevel(logging.ERROR)
logging.getLogger('pyutilib.component.core').setLevel(logging.ERROR)
logging.getLogger('pyutilib.component.core.pyomo').setLevel(logging.ERROR)
logging.getLogger('pyutilib.component.core.pca').setLevel(logging.ERROR)
logging.getLogger('Pyro4.core').setLevel(logging.ERROR)


def manhattan_dist(a, b):
    """Get the manhattan distance beteween two points 

    :param a: One point
    :type a: list
    :param b: Other point
    :type b: list
    :return: The manhattan distance between the two points
    :rtype: float
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def plan_milp(agent_pos, jobs, grid, config, plot=False):
    """Plans a scenario solutions using mixed-interger-linear programming.

    :param agent_pos: Positions of the agents
    :param jobs: Jobs start and goal positions
    :param grid:  The map to plan on
    :param config: Configuration dict
    :param plot: wether or not to plot results at the end
    :return: resulting agent to job allocations and resulting paths
    """
    # finding the assignment via MILP
    res_agent_job = optimize(agent_pos, jobs)

    # using our planner with prededined assignemt as MAPF planner
    _, _, res_paths = plan_cbsext(agent_pos, jobs, [], [], grid, config,
                                  plot=plot,
                                  pathplanning_only_assignment=res_agent_job)
    return res_agent_job, res_paths


def optimize(agents, tasks):
    """Run bonmin as MILP optimization to find a task allocation.

    :param agents: Agent positions
    :type agents: list
    :param tasks: Tasks with start / goal pairs
    :type tasks: list
    :return: Allocation that was found
    :rtype: list
    """
    # how long can consecutive tasks be at max?
    consec_len = len(tasks)
    # Precalculate distances
    # agents-tasks
    dist_at = np.zeros([len(agents), len(tasks)])
    for ia, it in product(range(len(agents)), range(len(tasks))):
        dist_at[ia, it] = manhattan_dist(agents[ia], tasks[it][0])
    # tasks
    dist_t = np.zeros([len(tasks)])
    for it in range(len(tasks)):
        dist_t[it] = manhattan_dist(tasks[it][0], tasks[it][1])
    # tasks-tasks
    dist_tt = np.zeros([len(tasks), len(tasks)])
    for it1, it2 in product(range(len(tasks)), range(len(tasks))):
        dist_tt[it1, it2] = manhattan_dist(tasks[it1][1], tasks[it2][0])

    # Problem
    m = ConcreteModel()

    # Sets
    def init_all(_):
        """
        Initalize a Set variable for all elements of the assignments

        :param _: We will not need data from the model here
        :return: The set
        """
        return ((a, c, t) for a in range(len(agents))
                for c in range(len(tasks))
                for t in range(len(tasks)))

    m.all = Set(dimen=3, initialize=init_all)

    def init_agents(_):
        return [a for a in range(len(agents))]

    m.agents = Set(dimen=1, initialize=init_agents)

    def init_tasks(_):
        return [t for t in range(len(tasks))]

    m.tasks = Set(dimen=1, initialize=init_tasks)

    def init_cons_a_first(_):
        return [t for t in range(1, len(tasks))]

    m.cons_a_first = Set(dimen=1, initialize=init_cons_a_first)

    # Variables
    m.assignments = Var(m.all,
                        within=NonNegativeIntegers)

    # Objective
    def total_duration(m):
        """
        Evaluating the sum of all jobs durations

        :type m: ConcreteModel
        :return: The objective
        """
        obj = 0
        for it in m.tasks:  # task we care about
            obj_task = 0
            for ia in range(len(agents)):  # for all agents
                path = 0
                path += m.assignments[ia, 0, it] * dist_at[ia][it]
                path += m.assignments[ia, 0, it] * dist_t[it]
                # we care for this path
                obj_task += path * m.assignments[ia, 0, it]
                for ic in range(1, len(tasks)):  # for all consecutive assignments
                    path += m.assignments[ia, ic, it] * \
                        path  # how did we get here?
                    for it2 in m.tasks:
                        for it2_prev in m.tasks:  # for all possible previous tasks
                            # from previous task end to this start
                            path += m.assignments[ia, ic, it2] * \
                                m.assignments[ia, ic - 1, it2_prev] * \
                                dist_tt[it2_prev][it2]
                        path += m.assignments[ia, ic, it] * dist_t[it]
                        # we care for this path
                        obj_task += path * m.assignments[ia, ic, it]
            obj += obj_task
        for it in m.tasks:
            # all tasks arrival time, TODO: whats the difference then?
            obj += tasks[it][2]
        return obj

    m.duration = Objective(rule=total_duration)

    # Constraints
    # every task has exactly one agent
    def one_agent_per_task(m, i_t):
        return sum(m.assignments[a, c, i_t] for a in m.agents for c in m.tasks) == 1

    m.one_agent = Constraint(m.tasks, rule=one_agent_per_task)

    # one agent can only have one task per time
    def one_task_per_time(m, i_a, i_c):
        return sum(m.assignments[i_a, i_c, t] for t in m.tasks) <= 1

    m.one_task = Constraint(m.agents, m.tasks, rule=one_task_per_time)

    # consecutive assignments can only happen after a previous one (consecutive)
    def consecutive(m, a, c):
        now = sum([m.assignments[a, c, t] for t in m.tasks])
        prev = sum([m.assignments[a, c - 1, t] for t in m.tasks])
        return now <= prev

    m.consec = Constraint(m.agents, m.cons_a_first, rule=consecutive)

    # Solve
    optim = SolverFactory('bonmin')
    result = optim.solve(m)

    # Solution

    # prob.display()

    agent_job = [tuple() for _ in m.agents]
    act = list(m.all)
    act.sort()
    for a, c, t in act:
        if m.assignments[(a, c, t)].value:
            agent_job[a] += (t,)

    return agent_job


if __name__ == "__main__":
    config = generate_config()
    config['filename_pathsave'] = ''
    agent_pos = [(1, 1), (9, 1), (3, 1)]  # three agents
    jobs = [((1, 6), (9, 6), 0),
            ((1, 3), (7, 3), 0),
            # ((6, 6), (3, 8), 10),
            # ((4, 8), (7, 1), 10),
            ((3, 4), (1, 5), 10)]
    grid = np.zeros([10, 10, 51])
    res = plan_milp(agent_pos, jobs, grid, config)
    print(res)

"""
/usr/bin/python3.6 /home/cch/src/miriam/planner/planner_milp_vs_cbsext.py
computation time: 76025.224703 s
((1,), (3, 2), (0,)) ((), (), ()) [([(0, 0, 0), (0, 1, 1), (0, 2, 2), (0, 3, 3), (0, 4, 4), (0, 5, 5), (0, 6, 6), (0, 7, 7), (0, 8, 8), (1, 8, 9)], [(1, 8, 10), (2, 8, 11), (3, 8, 12), (4, 8, 13), (5, 8, 14), (6, 8, 15), (7, 8, 16), (8, 8, 17), (8, 7, 18), (8, 6, 19), (7, 6, 20), (6, 6, 21), (5, 6, 22), (4, 6, 23), (3, 6, 24), (2, 6, 25), (2, 5, 26), (2, 4, 27)]), ([(3, 0, 0), (3, 1, 1), (3, 2, 2), (4, 2, 3), (5, 2, 4), (6, 2, 5), (7, 2, 6), (8, 2, 7), (8, 3, 8), (8, 4, 9), (8, 5, 10), (8, 6, 11), (8, 7, 12)], [(8, 7, 13), (8, 6, 14), (8, 5, 15), (8, 4, 16), (8, 3, 17), (8, 2, 18)], [(8, 2, 19), (8, 3, 20), (8, 4, 21), (8, 5, 22), (8, 6, 23), (7, 6, 24)], [(7, 6, 25), (8, 6, 26), (8, 7, 27), (8, 8, 28), (7, 8, 29), (6, 8, 30), (5, 8, 31), (4, 8, 32), (3, 8, 33)]), ([(2, 1, 0), (2, 2, 1), (1, 2, 2), (0, 2, 3), (0, 3, 4), (0, 4, 5), (0, 5, 6), (0, 6, 7), (0, 7, 8), (0, 8, 9)], [(0, 8, 10), (0, 7, 11), (0, 6, 12), (0, 5, 13), (0, 4, 14), (0, 3, 15), (0, 2, 16)])]

Process finished with exit code 0
"""
