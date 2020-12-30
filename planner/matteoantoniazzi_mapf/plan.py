from typing import List, Tuple, Any, Dict, Union

import numpy as np
from definitions import OBSTACLE

from .external.MAPFSolver.SearchBasedAlgorithms import ICTSSolver
from .external.MAPFSolver.Utilities import (Agent, Map, ProblemInstance,
                                            SolverSettings)


EXPANDED_NODES = 'expanded_nodes'


def icts(grid: np.ndarray, starts: np.ndarray, goals: np.ndarray
         ) -> Tuple[Any]:
    sse = SolverSettings()
    solver = ICTSSolver(sse)

    n_agents = starts.shape[0]
    obs_idx = np.where(grid == OBSTACLE)
    obstacles = list(
        map(lambda x: (obs_idx[0][x], obs_idx[1][x]), range(len(obs_idx[0]))))
    problem_map = Map(grid.shape[0], grid.shape[1], obstacles)
    agents = []
    for i in range(n_agents):
        agents.append(Agent(i, tuple(starts[i]), tuple(goals[i])))
    problem_instance = ProblemInstance(problem_map, agents)

    info = solver.solve(problem_instance, return_infos=True)
    return info


def expanded_nodes_from_info(info:
                             Tuple[List[List[Tuple[int, int]]],
                                   Dict[str, Union[int, float]]]) -> int:
    en = info[1][EXPANDED_NODES]
    assert isinstance(en, int)
    return en
