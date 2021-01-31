from typing import Any, Dict, List, Tuple, Union

import numpy as np
from definitions import OBSTACLE

from .external.MAPFSolver.SearchBasedAlgorithms import ICTSSolver
from .external.MAPFSolver.Utilities import (Agent, Map, ProblemInstance,
                                            SolverSettings)

EXPANDED_NODES = 'expanded_nodes'
SUM_OF_COSTS = 'sum_of_costs'
INFO_TYPE = Tuple[List[List[Tuple[int, int]]],
                  Dict[str, Union[int, float]]]


def icts_plan(grid: np.ndarray, starts: np.ndarray, goals: np.ndarray,
              timeout: int = 30) -> Tuple[Any]:
    sse = SolverSettings()
    sse.set_time_out(timeout)
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


def is_info_valid(info: INFO_TYPE) -> bool:
    return not len(info[0]) == 0


def expanded_nodes_from_info(info: INFO_TYPE) -> int:
    en = info[1][EXPANDED_NODES] + 1  # we count the root node also as expanded
    assert isinstance(en, int)
    return en


def sum_of_costs_from_info(info: INFO_TYPE) -> int:
    soc = info[1][SUM_OF_COSTS]
    assert isinstance(soc, int)
    return soc


def paths_from_info(info: INFO_TYPE) -> List[np.array]:
    paths = []
    for pt in info[0]:
        p = np.array(pt)
        le = p.shape[0]
        ts = np.arange(le)
        paths.append(
            np.append(p, ts.reshape((le, 1)), axis=1)
        )
    return paths
