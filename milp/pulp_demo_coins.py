from pulp import *
import random

# Example for IP to find smallest number of coins necessary to represent an amount of money in e.g. EUR coin sizes
# Author: Christian Henkel

prob = LpProblem("testCoins", LpMinimize)

# Variables
c1 = LpVariable("c1", 0, cat='Integer')
c2 = LpVariable("c2", 0, cat='Integer')
c5 = LpVariable("c5", 0, cat='Integer')
c10 = LpVariable("c10", 0, cat='Integer')
c20 = LpVariable("c20", 0, cat='Integer')
c50 = LpVariable("c50", 0, cat='Integer')
# Objective
prob += c1 + c2 + c5 + c10 + c20 + c50  # Amount of coins

# Constraints
prob += ((c1 * 1 +  # Values of coins
          c2 * 2 +
          c5 * 5 +
          c10 * 10 +
          c20 * 20 +
          c50 * 50) == 88)  # Target value

GLPK().solve(prob)

# Solution
for v in prob.variables():
    print(v.name, "=", v.varValue)

print("objective =", value(prob.objective))
