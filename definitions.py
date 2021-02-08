from typing import Tuple

import numpy as np

FREE = 0
INVALID = -1
OBSTACLE = 1

NO_SUCCESS = 0
SUCCESS = 1

# the time ecbs and icts etc are allowed to run for
DEFAULT_TIMEOUT_S = 30  # seconds;

SCENARIO_TYPE = Tuple[np.array, np.array, np.array]
