import itertools

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('bmh')

import pymc3 as pm


class state:
    nr_landmarks = 0
    iterations = 0

    durations = np.array([])
    durations_values = np.array([])
    last_timestamp = np.array([])

    mean_mu = np.array([])
    mean_sd = np.array([])
    std_mu = np.array([])
    std_sd = np.array([])


def update(_s: state, t: float, start: int, goal: int):
    """
    Update for new job
    :param _s: the state
    :param t: time the job occurred
    :param start: start landmark
    :param goal: goal landmark
    """
    _s.iterations += 1
    index = (int(start), int(goal))

    # Saving this timestamp
    _s.last_timestamp[index] = t

    # Save job in list
    for index_i in itertools.product(tuple(range(_s.nr_landmarks)),
                                     repeat=2): #  for all last starts and goals
        current_duration = t - _s.last_timestamp[index_i]
        if ((current_duration > 0) &
            (_s.last_timestamp[index_i] != -1)):
            index_save = (index_i[1], index[0])  # last goal + this start
            _s.durations_values[index_save] += 1
            _s.durations[:, index_save[0], index_save[1]] = np.roll(_s.durations[:, index_save[0], index_save[1]], 1)
            _s.durations[0, index_save[0], index_save[1]] = current_duration

    # Build model
    if ((_s.iterations > _s.durations.shape[0]) &
        (np.min(_s.durations_values[_s.durations_values>0]) > _s.durations.shape[0])):
        with pm.Model() as model:



    return _s


def update_list(_s: state, l: list):
    """
    Update for s list of jobs
    :param _s: the state
    :param l: list of jobs
    """
    for j in l:
        _s = update(_s, j[0], j[1], j[2])

    return _s


def info(_s: state):
    """
    Print info about current state
    :param _s: the state
    """
    print("=====")
    print("Iterations:", _s.iterations)
    print("-----")
    print("min durations", np.min(_s.durations))
    print("min durations[0]", np.min(_s.durations[0]))
    print("max durations", np.max(_s.durations))
    print("max durations[0,:,:]", np.max(_s.durations[0,:,:]))
    print("min durations_values", np.min(_s.durations_values))
    print("min durations_values[0]", np.min(_s.durations_values[0]))
    print("max durations_values", np.max(_s.durations_values))
    print("-----")



def init(n):
    """
    Initialize the state
    :param n: number of landmarks
    """
    window_length = 100
    _s = state
    _s.nr_landmarks = n
    _s.durations = np.zeros([window_length, n, n], dtype=float)
    _s.durations_values = np.zeros([n, n], dtype=int)
    _s.last_timestamp = np.zeros([n, n], dtype=float) - 1

    # model
    

    return _s


def estimation(s, start, goal):
    """
    retrieve one estimation 
    :param s: the state
    :param start: the start landmark to check
    :param goal: the goal
    """
    raise NotImplementedError()


if __name__ == "__main__":
    s = init(8)

    update(s, 0, 4, 5)

    print(estimation(s, 0, 0))
