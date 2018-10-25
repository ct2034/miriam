import png
import numpy as np
from itertools import *

r = png.Reader(filename='planner/tcbs/map.png')

x, y, iter, color = r.read()

m = np.vstack(map(np.sign, iter))
m = np.array(m, dtype=np.int8) -1

print(m)