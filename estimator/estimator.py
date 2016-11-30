import numpy as np


class state:
    t_mean = False
    t_std = False
    n = 0
    iterations = 0
    last = 0
    stat = []


def update(s, t, start, goal):
    """
    Update for new job
    :param s: the state
    :param t: time the job occurred
    :param start: start landmark
    :param goal: goal landmark
    """
    s.iterations += 1
    s.t_mean[start][goal] = t
    s.last = t


def update_list(s, l):
    """
    Update for s list of jobs
    :param s: the state
    :param l: list of jobs
    """
    window_size = s.n * 20
    for i, job in enumerate(l):
        iback = i
        seen = nb.zeros(s.n)
        while iback > 0 & nb.all(seen):
            if j[1] == 0:  # previous to 0
                print("to be implemented")

            iback -= 1


def info(s):
    """
    Print info about current state
    :param s: the state
    """
    print("to be implemented")


def init(n):
    """
    Initialize the state
    :param n: number of landmarks
    """
    s = state
    s.n = n
    s.t_mean = np.array(
        [[1 / n for i in range(n)] for i in range(n)]
    )
    s.t_std = np.array(
        [[999 for i in range(n)] for i in range(n)]
    )
    return state


def estimation(s, start, goal):
    """
    retrieve one estimation 
    :param s: the state
    :param start: the start landmark to check
    :param goal: the goal
    """
    return s.t_mean[start][goal], s.t_std[start][goal]


if __name__ == "__main__":
    s = init(8)

    update(s, 0, 4, 5)

    print(estimation(s, 0, 0))
