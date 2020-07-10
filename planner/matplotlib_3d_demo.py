import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from itertools import product

from tools import load_map

if __name__ == "__main__":
    _ = Axes3D
    plt.style.use('bmh')

    grid = load_map('map.png')
    grid = np.repeat(grid[:, :, np.newaxis], 100, axis=2)
    grid = np.swapaxes(grid, 0, 1)

    f = plt.figure()
    ax = f.add_subplot(111, projection='3d')
    ax.set_facecolor('white')
    ax.set_xlim3d(0, grid.shape[0])
    ax.set_ylim3d(0, grid.shape[1])
    ax.set_zlim3d(0, grid.shape[2])

    xx, yy = np.meshgrid(
        np.linspace(.5, grid.shape[0] + .5, grid.shape[0] * 50),
        np.linspace(.5, grid.shape[1] + .5, grid.shape[1] * 50))

    img = np.zeros([xx.shape[0], yy.shape[1]])

    for x, y in product(range(len(xx[0, :])), range(len(yy[:, 0]))):
        try:
            img[x, y] = grid[
                            int(round(xx[0, x] - 1)),
                            int(round(yy[y, 0] - 1)),
                            0] * -.001
        except IndexError:
            pass

    xx -= .5
    yy -= .5

    ax.contourf(xx, yy, img, z=-.2,
                antialiased=True, cmap=cm.Greys, alpha=0.8)

    plt.show()
