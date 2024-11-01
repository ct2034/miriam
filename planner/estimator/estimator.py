import itertools
import collections

import scipy
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("bmh")

import pymc3 as pm


class state:
    nr_landmarks = 0
    iterations = 0

    durations = np.array([])
    durations_values = np.array([])
    last_timestamp = np.array([])

    """Where in the durations have been zeros?"""
    zeros = np.array([])

    mean_mu = np.array([])
    mean_sd = np.array([])
    std_mu = np.array([])
    std_sd = np.array([])
    trace = False


def update(
    _s: state,
    t: float,
    start: int,
    goal: int,
    std_spread: float = 4,
    std_delta: float = 1,
):
    """
    Update for new job
    :param _s: the state
    :param t: time the job occurred
    :param start: start landmark
    :param goal: goal landmark
    :param std_spread: how to increase model params stds after each model update
    :param std_delta: how to increase model params stds after each model update
    """
    _s.iterations += 1
    index = (int(start), int(goal))

    # Saving this timestamp
    _s.last_timestamp[index] = t

    # Save job in list
    for index_i in itertools.product(
        tuple(range(_s.nr_landmarks)), repeat=2
    ):  # for all last starts and goals
        current_duration = t - _s.last_timestamp[index_i]
        if (current_duration > 0) & (_s.last_timestamp[index_i] != -1):
            index_save = (index_i[1], index[0])  # last goal + this start
            _s.durations_values[index_save] += 1
            _s.durations[:, index_save[0], index_save[1]] = np.roll(
                _s.durations[:, index_save[0], index_save[1]], 1
            )
            _s.durations[0, index_save[0], index_save[1]] = current_duration

    if (_s.iterations > _s.durations.shape[0]) & (np.max(_s.durations_values) > 0):
        if np.min(_s.durations_values[_s.durations_values > 0]) > _s.durations.shape[0]:
            n = _s.nr_landmarks

            # Removing zero values from durations
            durations_mean = np.mean(_s.durations[_s.durations > 0])
            durations_std = np.std(_s.durations[_s.durations > 0])

            _s.zeros = 0 == np.min(_s.durations, axis=0)
            for zero in itertools.product(tuple(range(_s.nr_landmarks)), repeat=2):
                if _s.zeros[zero]:
                    _s.durations[:, zero[0], zero[1]] = np.random.normal(
                        loc=durations_mean,
                        scale=durations_std,
                        size=_s.durations.shape[0],
                    )

            # Build model
            with pm.Model() as model:
                timing(True)
                mean = pm.Normal("mean", mu=_s.mean_mu, sd=_s.mean_sd, shape=(n, n))

                std = pm.Normal("std", mu=_s.std_mu, sd=_s.std_sd, shape=(n, n))

                Y = pm.Normal("Y", mu=mean, sd=std, observed=_s.durations)

                start = pm.find_MAP()
                print("found start")
                timing()

                step = pm.NUTS()

                _s.trace = pm.sample(500, step, start=start, progressbar=True)
                print("sample is_finished")
                timing()

                info(_s, True)
                # Evaluate results
                # TODO: before saving results, check for correct correlation
                mean_trace = _s.trace.get_values("mean")
                std_trace = _s.trace.get_values("std")

                burnin = int(0.3 * len(mean_trace))

                _s.mean_mu = fix_neg_values(np.mean(mean_trace[burnin:, :, :], axis=0))
                _s.mean_sd = (
                    fix_neg_values(np.std(mean_trace[burnin:, :, :], axis=0))
                    + std_delta
                ) * std_spread
                assert np.shape(_s.mean_mu) == (n, n), "Wrong Dimensions"

                _s.std_mu = fix_neg_values(np.mean(std_trace[burnin:, :, :], axis=0))
                _s.std_sd = (
                    fix_neg_values(np.std(std_trace[burnin:, :, :], axis=0)) + std_delta
                ) * std_spread
    return _s


def fix_neg_values(_array):
    _array[_array < 0] = 0
    return _array


def update_list(_s: state, l: list):
    """
    Update for s list of jobs
    :param _s: the state
    :param l: list of jobs
    """
    for j in l:
        _s = update(_s, j[0], j[1], j[2])

    return _s


def info(_s: state, plot: bool = False):
    """
    Print info about current state
    :param _s: the state
    :param plot: whether to plot something
    """
    print("=====")
    print("Iterations:", _s.iterations)
    print("-----")
    print("min durations", np.min(_s.durations))
    print("min durations[0]", np.min(_s.durations[0]))
    print("max durations", np.max(_s.durations))
    print("max durations[0,:,:]", np.max(_s.durations[0, :, :]))
    print("min durations_values", np.min(_s.durations_values))
    print("min durations_values[0]", np.min(_s.durations_values[0]))
    print("max durations_values", np.max(_s.durations_values))
    print("-----")

    pm_params = {"trace": _s.trace, "varnames": ["mean", "std"]}

    pm.summary(**pm_params)

    if plot:
        pm.traceplot(**pm_params)
        pm.autocorrplot(**pm_params)

    print("=====\n\n")


def init(n):
    """
    Initialize the state
    :param n: number of landmarks
    """
    window_length = 8 * n
    _s = state
    _s.nr_landmarks = n
    _s.durations = np.zeros([window_length, n, n], dtype=float)
    _s.durations_values = np.zeros([n, n], dtype=int)
    _s.last_timestamp = np.zeros([n, n], dtype=float) - 1

    # model
    _s.mean_mu = np.ones([n, n]) * 30
    _s.mean_sd = np.ones([n, n]) * 5
    _s.std_mu = np.ones([n, n]) * 4
    _s.std_sd = np.ones([n, n]) * 2

    return _s


def estimation(s, start, goal):
    """
    retrieve one estimation
    :param s: the state
    :param start: the start landmark to check
    :param goal: the goal
    """
    raise NotImplementedError()


last = False


def timing(reset=False):
    from datetime import datetime

    global last
    global start
    if reset or not last:
        last = datetime.now()
        start = datetime.now()
    else:
        duration = datetime.now() - last
        print("last:", duration.total_seconds(), "s")
        print("total:", (datetime.now() - start).total_seconds(), "s")
        last = datetime.now()


if __name__ == "__main__":
    s = init(8)

    update(s, 0, 4, 5)

    print(estimation(s, 0, 0))
