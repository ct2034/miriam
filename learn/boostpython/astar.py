#!/usr/bin/env python3

import libastar
import sys
sys.path.append('$HOME/src/miriam/learn/boostpython/build')

m = libastar.random_maze(10, 10)
print(">>>" + m.to_string())
print(m.solve())
print(m)
