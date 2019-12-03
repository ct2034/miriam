#!/usr/bin/env python2
import random
import networkx as nx
import numpy as np

from adamsmap_eval.eval_disc import synchronize_paths, eval_disc


def test_eval_disc_basic():
    # basic example
    g = nx.DiGraph()
    g.add_nodes_from(range(5))
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(1, 2)
    g.add_edge(1, 4)
    g.add_edge(2, 3)
    g.add_edge(2, 4)
    g.add_edge(3, 0)
    g.add_edge(4, 3)
    posar = [
        [1, 3],
        [3, 3],
        [2, 2],
        [1, 1],
        [3, 1]
    ]
    batch = np.array(
        [[0, 1], [3, 4]]
    )
    agent_diameter = 1
    v = .1
    eval_disc(batch, g, posar, agent_diameter, v)


def test_eval_disc_no_path():
    # a test where one job has definitely no path
    g = nx.DiGraph()
    g.add_nodes_from(range(3))
    g.add_edge(0, 1)
    g.add_edge(1, 0)
    posar = [
        [1, 3],
        [3, 3],
        [2, 2],
    ]
    batch = np.array(
        [[0, 1], [1, 2]]
    )
    agent_diameter = 1
    v = .1
    t, paths = eval_disc(batch, g, posar, agent_diameter, v)
    print(t)
    print(paths)


def test_synchronize_paths_basic():
    # basic example
    in_paths = [[1, 2, 3], [3, 2, 1]]
    expected1 = [[1, 1, 2, 3], [3, 2, 1]] # agent 1 has prio over 0
    expected2 = [[1, 2, 3], [3, 3, 2, 1]] # agent 0 has prio over 1
    out_paths = synchronize_paths(in_paths)
    assert expected1 == out_paths or expected2 == out_paths


def test_synchronize_paths_empty():
    # test if empty paths are handled correctly
    in_paths = [[1, 2, 3], [], [4, 5, 6]]
    expected = list(in_paths)
    out_paths = synchronize_paths(in_paths)
    assert expected == out_paths


def test_synchronize_paths_none():
    # test if None paths are handled correctly
    in_paths = [[1, 2, 3], None, [4, 5, 6], None]
    expected = list(in_paths)
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
    expected = list(in_paths)
    out_paths = synchronize_paths(in_paths)
    assert expected == out_paths


def test_synchronize_paths_rand_path():
    # test if the path is still correct
    in_paths, n_agents = generate_data()
    in_paths_undupl = []
    # make versions of teh data that do not contain the same vertex twice in succession
    for i_a in range(n_agents):
        tmp_path = []
        prev = -1
        for v in in_paths[i_a]:
            if v != prev:
                tmp_path.append(v)
                prev = v
        in_paths_undupl.append(tmp_path)
    out_paths = synchronize_paths(in_paths_undupl)
    # test if the out-paths are the same if we remove the duplications caused by synchronization
    for i_a in range(n_agents):
        tmp_path = []
        prev = -1
        for v in out_paths[i_a]:
            if v != prev:
                tmp_path.append(v)
                prev = v
        assert tmp_path == in_paths_undupl[i_a]
    # test if there is a collision in the out_paths
    T = max(map(len, out_paths))
    for t in range(T):
        occupied = set()
        for i_a in range(n_agents):
            if t < len(out_paths[i_a]):
                assert out_paths[i_a][t] not in occupied
                occupied.add(out_paths[i_a][t])


def generate_data():
    n_agents = 3
    length = 10
    in_paths = [
        [start] + [random.randint(0, 10) for _ in range(length - 1)]
        for start in range(n_agents)
    ]
    return in_paths, n_agents
