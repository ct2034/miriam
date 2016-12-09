import matplotlib.pyplot as plt
import numpy as np
import timeit

from astar import astar_grid8con

map = np.zeros([100, 100])
map[:80, 20] = -1
map[80, 20:60] = -1
map[20, 40:80] = -1
map[20:, 80] = -1

t = timeit.Timer()
t.timeit()

path = astar_grid8con.astar_grid8con((10, 10), (90, 90), map)

print("computation time:", t.repeat(), "s")

print("length: ", astar_grid8con.path_length(path))

fig, ax = plt.subplots()

ax.imshow(map.T, cmap='Greys', interpolation='nearest')
ax.set_title('astar path')
ax.axis([0, map.shape[0], 0, map.shape[1]])
ax.plot(
    np.array(np.matrix(path)[:, 0]).flatten(),
    np.array(np.matrix(path)[:, 1]).flatten(),
    c='b',
    lw=2
)

plt.show()
