from pulp import *

# Author: Christian Henkel

# Problem
prob = LpProblem("vector", LpMinimize)

# Variables
ax = LpVariable("ax", 0)
ay = LpVariable("ay", 0)
az = LpVariable("az", 0)

# Objective
prob += (ax + ay + az)  # small vector

# Constraints
prob += ((ax + ay + az) == 1)  # unity

prob += (ax == ay)  # middle in x-y pane
prob += (ax == az)  # ??

# Solve
GLPK().solve(prob)

# Solution
for v in prob.variables():
    print(v.name, "=", v.varValue)
print("objective =", value(prob.objective))
