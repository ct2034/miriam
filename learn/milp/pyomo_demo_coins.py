from __future__ import division

import numpy as np
from pyomo.core.base import Var
from pyomo.environ import (
    ConcreteModel,
    Constraint,
    Expression,
    NonNegativeIntegers,
    Objective,
)
from pyomo.opt import SolverFactory

model = ConcreteModel()

model.ct = Var(range(6), domain=NonNegativeIntegers)
values = [1, 2, 5, 10, 20, 50]


def obj_expression(model):
    return sum(model.ct)


model.OBJ = Objective(expr=obj_expression(model))


def _e(model):
    return np.dot(values, list(model.ct))


model.e = Expression([0], rule=_e)


def constr(model):
    return model.e == 88


model.Constraint1 = Constraint(expr=constr(model))


def ObjRule(model):
    return 2 * model.x[1] + 3 * model.x[2]


model.g = Objective(rule=ObjRule)

prob = model.create()
optim = SolverFactory("glpk")
result = optim.solve(prob, tee=True)
prob.load(result)

# all variables
prob.display()
