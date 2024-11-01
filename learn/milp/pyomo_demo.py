from __future__ import division

from pyomo.environ import *
from pyomo.opt import SolverFactory

model = ConcreteModel()

model.x = Var([1, 2], domain=NonNegativeReals)

model.OBJ = Objective(expr=2 * model.x[1] + 3 * model.x[2])

model.Constraint1 = Constraint(expr=3 * model.x[1] + 4 * model.x[2] >= 1)
prob = model.create_instance()
optim = SolverFactory("glpk")
SolverFactory
result = optim.solve(prob, tee=True)
prob.load(result)

# all variables
prob.display()
