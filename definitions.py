from typing import Any, List, Set, Tuple, TypeVar

import numpy as np
from libpysal.cg.kdtree import DISTANCE_METRICS

FREE = 0
INVALID = -1
OBSTACLE = 1

NO_SUCCESS = 0
SUCCESS = 1

# the time ecbs and icts etc are allowed to run for
DEFAULT_TIMEOUT_S = 30  # seconds;

SCENARIO_TYPE = Tuple[np.ndarray, np.ndarray, np.ndarray]
SCENARIO_RESULT = Tuple[float, float, float, float, Any]

# Indices of results
IDX_AVERAGE_TIME = 0
IDX_MAX_TIME = 1
IDX_AVERAGE_LENGTH = 2
IDX_MAX_LENGTH = 3
IDX_SUCCESS = 4

C = int
C_grid = Tuple[int, int]
N = Tuple[int, int]

EDGE_TYPE = Tuple[C, C, int]
BLOCKED_EDGES_TYPE = Set[EDGE_TYPE]
BLOCKED_NODES_TYPE = Set[N]
PATH = List[C]

# strings for graph attributes
POS = 'pos'
DISTANCE = 'distance'
