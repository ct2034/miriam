#!/usr/bin/env python3
import random

from eval_disc import synchronize_paths


def test_synchronize_paths_basic():
    # basic example
    in_paths = [[1, 2, 3], [3, 2, 1]]
    expected = [[1, 1, 2, 3], [3, 2, 1]]
    out_paths = synchronize_paths(in_paths)
    assert expected == out_paths


def test_synchronize_paths_empty():
    # test if empty paths are handled correctly
    in_paths = [[1, 2, 3], [], [4, 5, 6]]
    expected = in_paths.copy()
    out_paths = synchronize_paths(in_paths)
    assert expected == out_paths


def test_synchronize_paths_independent():
    # test if independent paths are handled correctly
    def rotate(l, n):
        l = list(l)
        return l[-n:] + l[:-n]

    length = 10
    in_paths = [
        list(range(length)),
        rotate(range(length), 1),
        rotate(range(length), 2),
        rotate(range(length), 3),
        rotate(range(length), -3),
        rotate(range(length), -2),
        rotate(range(length), -1)
    ]
    expected = in_paths.copy()
    out_paths = synchronize_paths(in_paths)
    assert expected == out_paths


def test_synchronize_paths_rand_collision():
    # test if there is a collision in the paths
    in_paths, n_agents = generate_data()
    out_paths = synchronize_paths(in_paths)
    T = max(map(len, out_paths))
    for t in range(T):
        occupied = set()
        for i_a in range(n_agents):
            if t < len(out_paths[i_a]):
                assert out_paths[i_a][t] not in occupied
                occupied.add(out_paths[i_a][t])


def test_synchronize_paths_rand_path():
    # test if the path is still correct
    in_paths, n_agents = generate_data()
    in_paths_undupl = []
    for i_a in range(n_agents):
        tmp_path = []
        prev = -1
        for v in in_paths[i_a]:
            if v != prev:
                tmp_path.append(v)
                prev = v
        in_paths_undupl.append(tmp_path)
    out_paths = synchronize_paths(in_paths_undupl)
    for i_a in range(n_agents):
        tmp_path = []
        prev = -1
        for v in out_paths[i_a]:
            if v != prev:
                tmp_path.append(v)
                prev = v
        assert tmp_path == in_paths_undupl[i_a]


def generate_data():
    n_agents = 3
    length = 10
    in_paths = [
        [start] + [random.randint(0, 10) for _ in range(length - 1)]
        for start in range(n_agents)
    ]
    return in_paths, n_agents
