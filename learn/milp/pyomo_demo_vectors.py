from __future__ import division
from pyomo.environ import *
from pyomo.opt import SolverFactory
import numpy as np

model = ConcreteModel()

model.a = Var(range(3), domain=Reals, bounds=(0, 1))
model.b = Var(range(3), domain=Reals)
model.c = Var(range(3), domain=Reals)


def ObjRule(model):
    return model.a[0] - .3


model.g = Objective(rule=ObjRule)

model.same_z_c = Constraint(expr=model.a[2] == model.b[2])
model.same_z_c2 = Constraint(expr=model.a[2] == model.c[2])

model.right_angle_c = Constraint(expr=model.a[0] * model.b[0] + model.a[1] * model.b[1] == - model.a[2] * model.b[2])
model.right_angle_c2 = Constraint(expr=model.a[0] * model.c[0] + model.a[1] * model.c[1] == - model.a[2] * model.c[2])

prob = model.create()
optim = SolverFactory('glpk')
result = optim.solve(prob, tee=True)
prob.load(result)

# all variables
prob.display()
