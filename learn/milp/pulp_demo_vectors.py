from pulp import *

# Author: Christian Henkel

# Problem
prob = LpProblem("vector", LpMinimize)

# Variables
a = [LpVariable("a" + str(i), 0) for i in range(3)]

# Objective
prob += lpSum(a)  # small vector

# Constraints
prob += ((lpSum(a)) == 1)  # unity

prob += (a[0] == a[1])  # middle in x-y pane
prob += (a[0] == a[2])  # ??

# Solve
GLPK().solve(prob)

# Solution
for v in prob.variables():
    print(v.name, "=", v.varValue)
print("objective =", value(prob.objective))
