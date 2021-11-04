import os
import pickle as pkl
from enum import Enum, auto
from typing import Any, Dict, OrderedDict

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
    DECEN = auto()


def get_filepath(scenario: SCENARIO_TYPE) -> str:
    # What path will this scenario be saved under?
    assert SCENARIO_STORAGE_PATH_ENVVAR_STR in os.environ
    folder = os.environ.get(SCENARIO_STORAGE_PATH_ENVVAR_STR)
    assert folder is not None
    (env, starts, goals) = scenario
    args = [env, starts, goals]
    fname = str(hasher(args, {})) + ".pkl"
    return folder + "/" + fname


def has_file(scenario: SCENARIO_TYPE) -> bool:
    # Is there a file for this scenario?
    return os.path.isfile(get_filepath(scenario))


def to_key_string(
        result_type: ResultType, solver_params: Dict[str, Any]) -> str:
    key_str: str = str(result_type.name).upper()
    solver_param_keys = sorted(solver_params.keys())
    for k in solver_param_keys:
        key_str += "_"
        key_str += str(k)
        key_str += "="
        key_str += str(solver_params[k])
    return key_str


def has_result(scenario: SCENARIO_TYPE, result_type: ResultType,
               solver_params: Dict[str, Any]) -> bool:
    # For this `scenario` is there a `result` in storage?
    key_str = to_key_string(result_type, solver_params)
    if has_file(scenario):
        with open(get_filepath(scenario), 'rb') as f:
            data = pkl.load(f)
            assert isinstance(data, dict)
            return key_str in data.keys()
    else:
        return False


def get_result(scenario: SCENARIO_TYPE, result_type: ResultType,
               solver_params: Dict[str, Any]) -> Any:
    # Retrieve `result` for `scenario`.
    key_str = to_key_string(result_type, solver_params)
    assert has_result(scenario, result_type, solver_params)
    with open(get_filepath(scenario), 'rb') as f:
        data = pkl.load(f)
        return data[key_str]


def save_result(scenario: SCENARIO_TYPE, result_type: ResultType,
                solver_params: Dict[str, Any], result: Any):
    # Save a `result` for `scenario`.
    key_str = to_key_string(result_type, solver_params)
    if has_file(scenario):
        with open(get_filepath(scenario), 'rb') as f:
            data: dict = pkl.load(f)
        data[key_str] = result
        with open(get_filepath(scenario), 'wb') as f:
            pkl.dump(data, f)
    else:
        with open(get_filepath(scenario), 'wb') as f:
            data = {key_str: result}
            pkl.dump(data, f)
