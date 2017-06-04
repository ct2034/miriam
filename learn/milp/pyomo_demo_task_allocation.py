from __future__ import division

import numpy as np
from itertools import *
from pyomo.environ import *
from pyomo.opt import SolverFactory


def manhattan_dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def optimize(agents, tasks):
    # Precalculate lengths
    # agents-tasks
    lengths_at = np.zeros([len(agents), len(tasks)])
    for ia in range(len(agents)):
        for it in range(len(tasks)):
            lengths_at[ia, it] = manhattan_dist(agents[ia], tasks[it][0])
    # tasks
    lengths_t = np.zeros([len(tasks)])
    for it in range(len(tasks)):
        lengths_t[it] = manhattan_dist(tasks[it][0], tasks[it][1])
    # tasks-tasks
    lengths_tt = np.zeros([len(tasks), len(tasks)])
    for it1 in range(len(tasks)):
        for it2 in range(len(tasks)):
            lengths_tt[it1, it2] = manhattan_dist(tasks[it1][1], tasks[it2][0])

    # Problem
    m = ConcreteModel()

    # Variables
    def init_n(_):
        return ((a, c, t) for a in range(len(agents)) for c in range(len(tasks)) for t in range(len(tasks)))
    m.n = Set(dimen=3, initialize=init_n)
    m.assignments = Var(m.n,
                        domain=Boolean)

    # Objective
    def total_duration(m):
        obj = 0
        for ia in range(len(agents)):  # for all agents
            # path to first task
            for it in np.arange(1, len(tasks)):
                obj += (m.assignments[ia, 0, it] * lengths_at[ia][it])
                obj += (m.assignments[ia, 0, it] * lengths_t[it])
            # for ic in np.arange(1, len(tasks)):
            #     # from previous task end to this start
            #     for it in np.arange(1, len(tasks)):
            #         temp = np.dot(m.assignments[ia + 1, ic], lengths_tt)
            #         obj += np.dot(m.assignments[ia + 1, ic + 1], temp)
            #         obj += np.dot(m.assignments[ia + 1, ic + 1], lengths_t)
        return obj
    m.duration = Objective(rule=total_duration)

    # Constraints
    # consecutive assignments only from beginning
    def one_agent_per_task(m):
        return m.assignments[1,0,1] == True
    m.one_agent = Constraint(rule=one_agent_per_task)

    # Solve
    prob = m.create_instance()
    optim = SolverFactory('glpk')
    result = optim.solve(prob, tee=True)

    # Solution
    prob.load(result)
    prob.display()


if __name__ == "__main__":
    agent_pos = [(1, 1), (9, 1), (3, 1)]  # three agents
    jobs = [((1, 6), (9, 6), 0), ((3, 3), (7, 3), 0), ((2, 4), (7, 8), 10)]
    optimize(agent_pos, jobs)