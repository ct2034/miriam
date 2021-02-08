from typing import Tuple

import numpy as np

FREE = 0
INVALID = -1
OBSTACLE = 1

DEFAULT_TIMEOUT_S = 30  # seconds; the time ecbs and icts etc are allowed to run for

SCENARIO_TYPE = Tuple[np.array, np.array, np.array]
