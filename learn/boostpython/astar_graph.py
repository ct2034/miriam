#!/usr/bin/env python3
from libastar_graph import AstarSolver

if __name__ == "__main__":
    a = AstarSolver()
    print(a.get())
    a.append(8)
    a.append(3)
    print(a.get())
