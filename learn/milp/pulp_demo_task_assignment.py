import numpy as np
from pulp import *

# Author: Christian Henkel


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
    prob = LpProblem("task_assignment", LpMinimize)

    # Variables
    assignments = []
    for ia in range(len(agents)):  # for all agents
        agent_temp = []
        for ic in range(len(tasks)):  # over possibly consecutive tasks
            consec_temp = []
            for it in range(len(tasks)):  # for all possible tasks
                consec_temp.append(
                    LpVariable(
                        "assignment_a%d_c%d_t%d" % (ia, ic, it),
                        lowBound=0,
                        cat=LpBinary,
                    )
                )
            agent_temp.append(consec_temp)
        assignments.append(agent_temp)

    # Objective
    for ia in range(len(agents)):  # for all agents
        # path to first task
        prob += lpDot(assignments[ia][0], lengths_at[ia])
        prob += lpDot(assignments[ia][0], lengths_t)
        for ic in np.arange(1, len(tasks)):
            # from previous task end to this start
            temp = lpDot(lengths_tt, assignments[ia][ic - 1])
            prob += lpDot(assignments[ia][ic], temp)
            prob += lpDot(assignments[ia][ic], lengths_t)

    # Constraints
    # consecutive assignments only from beginning
    prob += 0  # ?!?
    # only one agent per task
    prob += 0  # ?!?

    # Solve
    GLPK().solve(prob)

    # Solution
    for v in prob.variables():
        print(v.name, "=", v.varValue)
    print("objective =", value(prob.objective))


if __name__ == "__main__":
    agent_pos = [(1, 1), (9, 1), (3, 1)]  # three agents
    jobs = [((1, 6), (9, 6), 0), ((3, 3), (7, 3), 0), ((2, 4), (7, 8), 10)]
    optimize(agent_pos, jobs)
