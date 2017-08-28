import munkres
import numpy as np

from tools import benchmark, mongodb_save, get_system_parameters

m = munkres.Munkres()


def test_munkres_benchmark():
    user, _, _ = get_system_parameters()
    if user == 'travis':
        vals = [10, 30, 70, 100, 300, 500]
    else:
        vals = [10, 30, 70]
    ts = benchmark(
        lambda x: m.compute(np.random.rand(x, x)),
        vals
    )
    print(ts)
    mongodb_save(
        'test_munkres_benchmark', {
            'values': vals,
            'durations': ts
        }
    )

if __name__ == "__main__":
    matrix = [[1, 2, 3], [3, 1, 2], [2, 3, 1]]
    print(m.compute(matrix))

    matrix = [[1, 2, 3], [3, 1, 2], [2, 3, 10], [1, 1, 1], [10, 10, 10]]
    print(m.compute(matrix))

    test_munkres_benchmark()
