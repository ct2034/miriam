import math
from pulp import *
import random

# Author: Christian Henkel

COS120 = -.5

prob = LpProblem("testCoins", LpMinimize)

# Variables
ax = LpVariable("ax", 0)
ay = LpVariable("ay", 0)
az = LpVariable("az", 0)
bx = LpVariable("bx")
by = LpVariable("by")
bz = LpVariable("bz")
cx = LpVariable("cx")
cy = LpVariable("cy")
cz = LpVariable("cz")

prob += ax + ay + az == 1  # unity

prob += lpDot(ax, bx) + lpDot(ay, by) + lpDot(az, bz) == 0
prob += ax * cx + ay * cy + az * cz == 0
prob += cx * bx + cy * by + cz * bz == 0

prob += az - bz == 0
prob += az - cz == 0

GLPK().solve(prob)

# Solution
for v in prob.variables():
    print(v.name, "=", v.varValue)

print("objective =", value(prob.objective))
