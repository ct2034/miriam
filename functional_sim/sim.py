import numpy as np
from random import random


class state:
    agvs = False
    jobs = False


def init(n):
    s = state
    s.agvs = np.random.random([3, n])
    return s


def job(s, sx, sy, gx, gy):
    return s


def iterate(s):
    return s


if __name__ == "__main__":
    s = init(5)

    for i in range(1000):
        if i % 100 == 0:
            job(s, random(), random(), random(), random())
        iterate(s)
