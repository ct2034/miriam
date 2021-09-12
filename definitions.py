from typing import List, Set, Tuple, TypeVar

import numpy as np

FREE = 0
INVALID = -1
OBSTACLE = 1

NO_SUCCESS = 0
SUCCESS = 1

# the time ecbs and icts etc are allowed to run for
DEFAULT_TIMEOUT_S = 30  # seconds;

SCENARIO_TYPE = Tuple[np.ndarray, np.ndarray, np.ndarray]

C = TypeVar('C', Tuple[int, int], Tuple[int])  # location coordinate
N = TypeVar('N', Tuple[int, int, int], Tuple[int, int])  # planning node

EDGE_TYPE = Tuple[C, C, int]
BLOCKED_EDGES_TYPE = Set[EDGE_TYPE]
BLOCKED_NODES_TYPE = Set[N]
PATH = List[N]
