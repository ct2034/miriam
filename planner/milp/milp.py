from __future__ import division

import numpy as np
from itertools import *
from pyomo.environ import *
from pyomo.core import *
from pyomo.opt import SolverFactory


def manhattan_dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def optimize(agents, tasks):
    # how long can consecutive tasks be at max?
    consec_len = 1 + len(tasks) - len(agents)  # TODO: use this
    # Precalculate distances
    # agents-tasks
    dist_at = np.zeros([len(agents), len(tasks)])
    for ia in range(len(agents)):
        for it in range(len(tasks)):
            dist_at[ia, it] = manhattan_dist(agents[ia], tasks[it][0])
    # tasks
    dist_t = np.zeros([len(tasks)])
    for it in range(len(tasks)):
        dist_t[it] = manhattan_dist(tasks[it][0], tasks[it][1])
    # tasks-tasks
    dist_tt = np.zeros([len(tasks), len(tasks)])
    for it1 in range(len(tasks)):
        for it2 in range(len(tasks)):
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
        for ia in range(len(agents)):  # for all agents
            # path to first task
            obj_agent = 0
            for it in m.tasks:
                obj_agent += m.assignments[ia, 0, it] * dist_at[ia][it]
                obj_agent += m.assignments[ia, 0, it] * dist_t[it]
            for ic in range(1, len(tasks)):  # for all consecutive assignments
                obj_agent += m.assignments[ia, ic, it] * obj_agent  # how did we get here?
                for it in m.tasks:
                    for it_prev in m.tasks:  # for all possible previous tasks
                        # from previous task end to this start
                        obj_agent += m.assignments[ia, ic, it] * \
                                     m.assignments[ia, ic - 1, it_prev] * \
                                     dist_tt[it_prev][it]
                    obj_agent += m.assignments[ia, ic, it] * dist_t[it]
        for it in m.tasks:
            obj += tasks[it][2]  # all tasks arrival time, TODO: whats the difference then?
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
    prob = m.create_instance()
    optim = SolverFactory('cplex')
    result = optim.solve(prob, tee=True)

    # Solution
    prob.load(result)
    prob.display()


if __name__ == "__main__":
    agent_pos = [(1, 1), (9, 1), (3, 1)]  # three agents
    jobs = [((1, 6), (9, 6), 0),
            ((1, 3), (7, 3), 0),
            ((6, 6), (3, 8), 10),
            ((4, 8), (7, 1), 10),
            ((3, 4), (1, 5), 10)]
    optimize(agent_pos, jobs)
