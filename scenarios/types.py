# methods to check type of environment
from enum import Enum, auto
from typing import Dict, Tuple, Union

import networkx as nx
import numpy as np

POTENTIAL_ENV_TYPE = Union[np.ndarray, nx.Graph]

COORD_TO_NODE_TYPE = Dict[Tuple[int, int], int]


class EnvType(Enum):
    GRIDMAP = auto()
    ROADMAP = auto()


def _get_type(env: POTENTIAL_ENV_TYPE) -> EnvType:
    if isinstance(env, np.ndarray) and env.shape[0] == env.shape[1]:
        return EnvType.GRIDMAP
    elif isinstance(env, nx.Graph):
        return EnvType.ROADMAP
    else:
        raise RuntimeError(f"Could not determine environment type of {env} correctly.")


def is_gridmap(env: POTENTIAL_ENV_TYPE) -> bool:
    return EnvType.GRIDMAP == _get_type(env)


def is_roadmap(env: POTENTIAL_ENV_TYPE) -> bool:
    return EnvType.ROADMAP == _get_type(env)
