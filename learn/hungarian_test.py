import munkres
import numpy as np

from tools import benchmark, is_travis, mongodb_save

m = munkres.Munkres()


def munkres_call(x):
    res = m.compute(np.random.rand(x, x))
    print(res)
    return len(res)  # does not really make sense


def test_munkres_benchmark():
    if is_travis():
        vals = [10, 30, 70, 100, 300]
    else:
        vals = [10, 30]
    ts, res = benchmark(munkres_call, [vals], samples=5)
    print(ts)
    mongodb_save("test_munkres_benchmark", {"values": vals, "durations": ts.tolist()})


if __name__ == "__main__":
    matrix = [[1, 2, 3], [3, 1, 2], [2, 3, 1]]
    print(m.compute(matrix))

    matrix = [[1, 2, 3], [3, 1, 2], [2, 3, 10], [1, 1, 1], [10, 10, 10]]
    print(m.compute(matrix))

    test_munkres_benchmark()
