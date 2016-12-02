import itertools

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('bmh')


class state:
    last = {}
    nr_landmarks = 0
    iterations = 0

    testing = []
    legend = []


def update(_s: state, t: float, start: int, goal: int):
    """
    Update for new job
    :param _s: the state
    :param t: time the job occurred
    :param start: start landmark
    :param goal: goal landmark
    """
    _s.iterations += 1

    # TODO: this is only a test for one pair: now from 2, when last to all?
    l = []
    legend = []
    legend_set = False

    def from_to(_pair, l):
        if (_pair in _s.last.keys()):
            l.append(t - _s.last[_pair])
            if not legend_set:
                legend.append(str(_pair))
        return l

    if start == 2:  # TODO: assuming current testcase
        for pair in filter(lambda x: True, itertools.product(range(_s.nr_landmarks), repeat=2)):
            l = from_to(pair, l)

    if len(l) == _s.nr_landmarks:
        _s.testing.append(l)
        _s.legend = legend
        legend_set = True

    key = (int(start), int(goal))
    _s.last.update({key: t})

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


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """
    src: https://stackoverflow.com/questions/22988882/how-to-smooth-a-curve-in-python
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError as msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


def info(_s: state):
    """
    Print info about current state
    :param _s: the state
    """
    #smooth
    smooth = np.array(_s.testing)
    for i in range(_s.nr_landmarks):
        smooth[:, i] = savitzky_golay(np.array(_s.testing)[:, i], 23, 3)

    plt.plot(smooth)
    plt.legend(_s.legend)

    plt.show()


def init(n):
    """
    Initialize the state
    :param n: number of landmarks
    """
    _s = state
    _s.nr_landmarks = n
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
