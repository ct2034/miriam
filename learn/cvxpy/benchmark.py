import random
import timeit

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from learn.cvxpy.two_lines import min_dist

if __name__ == "__main__":
    n_tests = 100
    n_subtest = 100
    rng = random.Random(0)
    took_mean = []
    took_std = []

    for _ in tqdm(range(n_tests)):
        took_sub = []
        for _ in range(n_subtest):
            # 4 points
            pts = np.array([[rng.uniform(-1, 1), rng.uniform(-1, 1)] for _ in range(4)])

            l1 = pts[0:2, :]
            l2 = pts[2:4, :]

            t = timeit.timeit(lambda: min_dist(l1, l2), number=1)
            took_sub.append(t)
        took_mean.append(np.mean(took_sub))
        took_std.append(np.std(took_sub))

    took_mean = np.array(took_mean)
    took_std = np.array(took_std)

    plt.plot(took_mean)
    plt.fill_between(
        range(n_tests), took_mean - took_std, took_mean + took_std, alpha=0.5
    )

    plt.savefig("learn/cvxpy/benchmark.png")
