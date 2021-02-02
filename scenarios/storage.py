import os
import pickle as pkl
from enum import Enum, auto
from typing import Any

from definitions import SCENARIO_TYPE
from numpy.core.numerictypes import ScalarType
from tools import hasher

"""
This can store and recall saved scenario solutions for different planners.
Basic requirement is the SCENARIO_STORAGE_PATH environment variable to be set.
"""
SCENARIO_STORAGE_PATH_ENVVAR_STR = 'SCENARIO_STORAGE_PATH'


class ResultType(Enum):
    ECBS_PATHS = auto()
    ECBS_DATA = auto()
    ICTS_PATHS = auto()
    ICTS_INFO = auto()
    INDEP_PATHS = auto()


def get_filepath(scenario: SCENARIO_TYPE) -> str:
    # What path will this scenario be saved under?
    folder = os.environ.get(SCENARIO_STORAGE_PATH_ENVVAR_STR)
    assert folder is not None
    (env, starts, goals) = scenario
    kwargs = {"env": env,
              "starts": starts,
              "goals": goals}
    fname = str(hasher([], kwargs)) + ".pkl"
    return folder + "/" + fname


def has_file(scenario: SCENARIO_TYPE) -> bool:
    # Is there a file for this scenario?
    return os.path.isfile(get_filepath(scenario))


def has_result(scenario: SCENARIO_TYPE, result_type: ResultType) -> bool:
    # For this `scenario` is there a `result` in storage?
    if has_file(scenario):
        with open(get_filepath(scenario), 'rb') as f:
            data = pkl.load(f)
            assert isinstance(data, dict)
            return result_type.name in data.keys()
    else:
        return False


def get_result(scenario: SCENARIO_TYPE, result_type: ResultType) -> Any:
    # Retrieve `result` for `scenario`.
    with open(get_filepath(scenario), 'rb') as f:
        data = pkl.load(f)
        assert result_type.name in data.keys()
        return data[result_type.name]


def save_result(scenario: SCENARIO_TYPE, result_type: ResultType, result: Any):
    # Save a `result` for `scenario`.
    if has_file(scenario):
        with open(get_filepath(scenario), 'r+b') as f:
            data: dict = pkl.load(f)
            assert result_type.name not in data.keys()  # was not there before
            data[result_type.name] = result
            pkl.dump(result, f)
    else:
        with open(get_filepath(scenario), 'wb') as f:
            data = {result_type.name: result}
            pkl.dump(result, f)
