#!/usr/bin/env python3
from libastar_graph import AstarSolver

if __name__ == "__main__":
    pos = [
        [.1, .1],
        [.1, .9],
        [.9, .9],
        [.9, .1]
    ]
    edges = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0]
    ]

    a = AstarSolver(pos, edges)
    print(a)
    print(a.get(0).x)
    print(a.get(1).x)
    print(a.get(2).x)
    print(a.get(3).x)
    a.append(8)
    a.append(3)
