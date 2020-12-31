from __future__ import print_function
from ortools.linear_solver import pywraplp


def main(x1, x2, t):
    # Create the linear solver with the GLOP backend.
    solver = pywraplp.Solver('simple_lp_program',
                             pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

    # polynome
    #   x(t) = a_x x^3 + b_x x^2 + c_x x + d_x
    #   v(t) = 3a_x x^2 + 2b_x x + c_x
    #   a(t) = 6a_x x + 2b_x
    #   j(t) = 6a_x

    # THIS IS NOT LINEAR, IS IT?!

    # ....


if __name__ == '__main__':
    main(1, 2, 5)
