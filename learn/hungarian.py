import munkres
import numpy as np

from tools import benchmark

m = munkres.Munkres()


def test_munkres_benchmark():
    benchmark(
        lambda x: m.compute(np.random.rand(x, x)),
        [10, 30, 100, 300]
    )


if __name__ == "__main__":
    matrix = [[1, 2, 3], [3, 1, 2], [2, 3, 1]]
    print(m.compute(matrix))

    matrix = [[1, 2, 3], [3, 1, 2], [2, 3, 10], [1, 1, 1], [10, 10, 10]]
    print(m.compute(matrix))

    test_munkres_benchmark()
