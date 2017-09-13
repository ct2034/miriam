import numpy as np

from planner.astar.astar_grid48con import astar_grid4con
from planner.astar.base import NoPathException

VERTEX = 0
EDGE = 1

path_save = {}

def path(start: tuple, goal: tuple, _map: np.array, blocked: list, path_save_process: dict = {}, calc: bool = True):
    """
    Calculate or return pre-calculated path from start to goal

    Args:
      start: The start to start from
      goal: The goal to plan to
      _map: The map to plan on
      blocked: List of blocked points for agents e.g. ((x, y, t), agent)
      path_save_process: pre-processed paths are saved here
      calc: whether or not the path should be calculated if no saved id available. (returns False if not saved)

    Returns:
      the path as list of tuples
      or [] if no path found
      or False if path shouldn't have been calculated but was not saved either
    """
    seen = set()
    _map = _map.copy()
    for b in blocked:
        if b[0] == VERTEX:
            v = b[1]
            _map[(v[1],
                  v[0],
                  v[2])] = -1
        elif b[0] == EDGE:
            for v in b[1][0:2]:
                _map[(v[1],
                      v[0],
                      b[1][2])] = -1
                # TODO: actually block edge!

    blocked.sort()
    startgoal = [start, goal]
    index = tuple(startgoal) + tuple(blocked)
    global path_save
    if index not in path_save.keys():
        if calc:  # if we want to calc (i.e. find the cost)
            assert len(start) == 2, "Should be called with only spatial coords"
            try:
                _path = astar_grid4con(start + (0,),
                                       goal + (_map.shape[2] - 1,),
                                       _map.swapaxes(0, 1))
            except NoPathException:
                _path = []

            path_save_process[index] = _path
        else:
            return False, {}
    else:  # exists in the path_save
        _path = path_save[index]

    for b in blocked:
        if b[0] == VERTEX and b[1] in _path:
            return False, {}
            # TODO (maybe): test for edges?

    if not _path:
        return False, {}

    assert start == _path[0][0:2], "Planed path starts not from start"
    assert goal == _path[-1][0:2], "Planed path ends not in goal"
    return _path, path_save_process
