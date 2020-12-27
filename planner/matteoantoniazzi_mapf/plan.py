from .external.MAPFSolver.SearchBasedAlgorithms import ICTSSolver
from .external.MAPFSolver.Utilities import (Agent, ProblemInstance,
                                            SolverSettings)


def icts(grid, starts, goals):
    sse = SolverSettings()
    ICTSSolver(sse)
