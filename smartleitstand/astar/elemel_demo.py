import timeit

elemel_astar = __import__("elemel_python-astar.src.astar")
import matplotlib.pyplot as plt
import numpy as np

from smartleitstand.astar import astar_grid48con

grid = np.zeros([100, 100, 100])
grid[:80, 20, :] = -1
grid[80, 20:60, :] = -1
grid[20, 40:80, :] = -1
grid[20:, 80, :] = -1

start = (10, 10, 0)
goal = (90, 90, 99)

t = timeit.Timer()
t.timeit()
path = elemel_astar.src.astar.astar(start_pos=start,
                                    neighbors=lambda pos: astar_grid48con.get_children(pos, grid),
                                    goal=lambda pos: pos == goal,
                                    start_g=0,
                                    cost=lambda a, b: astar_grid48con.cost(a, b, False),
                                    heuristic=lambda pos: astar_grid48con.heuristic(pos, goal, False)
                                    )
print("computation time:", t.repeat(), "s")

# make comparable:
path.insert(0, start)

print("length: ", astar_grid48con.path_length(path))

fig, ax = plt.subplots()

# ax.imshow(grid.T, cmap='Greys', interpolation='nearest')
ax.set_title('astar path')
ax.axis([0, grid.shape[0], 0, grid.shape[1]])
ax.plot(
    np.array(np.matrix(path)[:, 0]).flatten(),
    np.array(np.matrix(path)[:, 1]).flatten(),
    c='b',
    lw=2
)

plt.show()
